#!/usr/bin/env python3
"""Run a host-side inference pass that mirrors the Vitis KWS deployment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required to run this utility. Install it with `pip install torch`."
    ) from exc

try:
    import torchaudio
    import torchaudio.transforms as T
except ImportError as exc:
    raise SystemExit(
        "torchaudio is required to load WAV files. Install it with `pip install torchaudio`."
    ) from exc

TARGET_SAMPLE_RATE = 16000
NUM_MELS = 40
WINDOW_SIZE = 400
HOP_LENGTH = 160
NUM_FRAMES = 98
FFT_SIZE = WINDOW_SIZE
FFT_BINS = FFT_SIZE // 2 + 1
LOG_EPS = 1e-6
INT32_SCALE = 2147483648.0

_HANN = np.hanning(WINDOW_SIZE).astype(np.float32)
_COS_TABLE = np.cos(
    2.0
    * math.pi
    * np.arange(FFT_BINS, dtype=np.float32)[:, None]
    * np.arange(WINDOW_SIZE, dtype=np.float32)[None, :]
    / float(FFT_SIZE)
).astype(np.float32)
_SIN_TABLE = np.sin(
    2.0
    * math.pi
    * np.arange(FFT_BINS, dtype=np.float32)[:, None]
    * np.arange(WINDOW_SIZE, dtype=np.float32)[None, :]
    / float(FFT_SIZE)
).astype(np.float32)


def _mel_filterbank(num_mels: int, fft_bins: int, sample_rate: int) -> np.ndarray:
    mel_min = 2595.0 * math.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, num_mels + 2, dtype=np.float32)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_points = np.floor((FFT_SIZE + 1) * hz_points / sample_rate).astype(np.int32)
    bin_points = np.clip(bin_points, 0, fft_bins - 1)

    filters = np.zeros((num_mels, fft_bins), dtype=np.float32)
    for m in range(1, num_mels + 1):
        start = bin_points[m - 1]
        center = bin_points[m]
        end = bin_points[m + 1]
        if center <= start:
            center = start + 1
        if end <= center:
            end = center + 1

        up_range = center - start
        if up_range <= 0:
            continue
        filters[m - 1, start:center] = (
            np.arange(start, center, dtype=np.float32) - float(start)
        ) / float(up_range)

        down_range = end - center
        if down_range <= 0:
            continue
        filters[m - 1, center:end] = (
            float(end) - np.arange(center, end, dtype=np.float32)
        ) / float(down_range)

    return filters


_MEL_FILTER = _mel_filterbank(NUM_MELS, FFT_BINS, TARGET_SAMPLE_RATE)


class SignActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = x.sign()
        out[out == 0] = -1
        return out


class _BinaryWeight(nn.Module):
    def __init__(self, shape: torch.Size) -> None:
        super().__init__()
        self.register_buffer("weight_bin", torch.ones(shape))


class QConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.activation = SignActivation()
        self.conv = _BinaryWeight(torch.Size([out_channels, in_channels, kernel_size, kernel_size]))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.activation(x)
        weight = self.conv.weight_bin
        return F.conv2d(x, weight, None, stride=self.stride, padding=self.padding)


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.register_buffer("weight_bin", torch.ones(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return F.linear(x, self.weight_bin, None)


class BNNKWS(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = QConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = QConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = QConv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = QConv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.act_out = SignActivation()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = BinaryLinear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.conv1(x) + 1.0
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.act_out(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def waveform_to_logmel(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)(waveform)

    target_len = TARGET_SAMPLE_RATE
    cur_len = waveform.shape[-1]
    if cur_len < target_len:
        waveform = F.pad(waveform, (0, target_len - cur_len))
    elif cur_len > target_len:
        start = (cur_len - target_len) // 2
        waveform = waveform[:, start : start + target_len]

    mono = waveform.squeeze(0).cpu().numpy().astype(np.float32)
    features = np.zeros((NUM_MELS, NUM_FRAMES), dtype=np.float32)

    for frame in range(NUM_FRAMES):
        frame_offset = frame * HOP_LENGTH
        window = mono[frame_offset : frame_offset + WINDOW_SIZE]
        if window.shape[0] < WINDOW_SIZE:
            padded = np.zeros(WINDOW_SIZE, dtype=np.float32)
            padded[: window.shape[0]] = window
            window = padded

        windowed = window * _HANN
        real = _COS_TABLE @ windowed
        imag = -_SIN_TABLE @ windowed
        power = real * real + imag * imag

        mel = _MEL_FILTER @ power
        mel[mel < LOG_EPS] = LOG_EPS
        features[:, frame] = np.log(mel)

    return torch.from_numpy(features)


DEFAULT_LABELS: Sequence[str] = (
    "yes",
    "no",
    "go",
    "on",
    "wow",
    "happy",
    "follow",
    "off",
    "stop",
    "visual",
)


def _load_labels(path: Path | None) -> Sequence[str]:
    if path is None:
        return DEFAULT_LABELS
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            try:
                indices = sorted((int(k), v) for k, v in data.items())
                return [label for _, label in indices]
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid label mapping in {path}: {exc}")
        if isinstance(data, list):
            return data
        raise SystemExit(f"Unsupported JSON structure for labels in {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"Label file {path} is empty")
    return lines


def _load_waveform_from_wav(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform, sample_rate


def _load_waveform_from_bin(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    if not raw:
        raise SystemExit(f"Audio binary {path} is empty")
    if len(raw) % 4 != 0:
        raise SystemExit(f"Audio binary {path} size is not a multiple of 4 bytes")
    samples = []
    for i in range(0, len(raw), 4):
        sample = int.from_bytes(raw[i : i + 4], "little", signed=True)
        samples.append(sample)
    waveform = torch.tensor(samples, dtype=torch.float32) / INT32_SCALE
    return waveform.unsqueeze(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BNN KWS model on a single clip")
    parser.add_argument("weights", type=Path, help="Path to the PyTorch checkpoint (.pt)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path, help="Path to the WAV file to classify")
    group.add_argument("--bin", type=Path, help="Path to a little-endian int32 PCM binary clip")
    parser.add_argument("--labels", type=Path, help="Optional label list (text or JSON)")
    parser.add_argument("--topk", type=int, default=5, help="How many top classes to display")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--dump-logits", action="store_true", help="Print raw logits as well")
    return parser.parse_args()


def build_model(state_dict: dict) -> BNNKWS:
    if isinstance(state_dict, dict) and "model" in state_dict:
        raw_state = state_dict["model"]
        num_classes = int(
            state_dict.get("num_classes")
            or state_dict.get("args", {}).get("num_classes", 0)
            or 0
        )
    else:
        raw_state = state_dict
        num_classes = 0

    if not isinstance(raw_state, dict):
        raise SystemExit("Checkpoint must be a mapping of parameter names to tensors")

    if not num_classes:
        fc_key = next(
            (key for key in raw_state if key.endswith("fc.weight_bin") or key.endswith("fc.weight")),
            None,
        )
        if fc_key is None:
            raise SystemExit("Unable to infer number of classes from checkpoint")
        num_classes = int(raw_state[fc_key].shape[0])

    model = BNNKWS(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(raw_state, strict=False)
    if unexpected:
        raise SystemExit(f"Unexpected keys in checkpoint: {unexpected}")
    if missing:
        raise SystemExit(f"Missing parameters when loading checkpoint: {missing}")
    model.eval()
    return model


def run_inference() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ckpt = torch.load(str(args.weights), map_location=device)
    model = build_model(ckpt).to(device)

    if args.wav is not None:
        waveform, sample_rate = _load_waveform_from_wav(args.wav)
    else:
        waveform = _load_waveform_from_bin(args.bin)
        sample_rate = TARGET_SAMPLE_RATE

    features = waveform_to_logmel(waveform, sample_rate).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(features)
        probs = logits.softmax(dim=1).squeeze(0)

    labels = _load_labels(args.labels)
    if len(labels) != probs.numel():
        raise SystemExit(
            f"Label count ({len(labels)}) does not match model output ({probs.numel()})"
        )

    pred_idx = int(probs.argmax().item())
    print(f"Predicted index: {pred_idx}")
    print(f"Predicted label: {labels[pred_idx]}")

    topk = min(max(args.topk, 1), probs.numel())
    values, indices = torch.topk(probs, topk)
    print("Top classes:")
    for rank, (idx, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f"  {rank:>2}: {labels[idx]:<10} ({idx:>2}) -> {value * 100:.2f}%")

    if args.dump_logits:
        print("\nLogits:")
        for idx, value in enumerate(logits.squeeze(0).tolist()):
            print(f"  {idx:>2}: {value:.6f}")


if __name__ == "__main__":
    run_inference()

#!/usr/bin/env python3
"""Run a host-side inference pass that mirrors the Vitis KWS deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
LOG_EPS = 1e-6
INT32_SCALE = 2147483648.0


class SignSTE(torch.autograd.Function):
    """Straight-through estimator for the sign activation used in the BNN."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x.sign().clamp(min=-1.0, max=1.0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return grad_output


def binarize(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class BinaryActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return binarize(x)


class BinaryConv2d(nn.Module):
    """Conv2d layer whose weights are binarised at inference time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        use_scale: bool = True,
    ) -> None:
        super().__init__()
        self.real_weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.real_weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_scale = use_scale
        self.act = BinaryActivation()
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        weight, alpha = weight_binarize(self.real_weight, self.use_scale)
        if alpha is not None:
            weight = weight * alpha
        x = F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)
        x = self.bn(x)
        x = self.act(x)
        return x


class BinaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, use_scale: bool = True) -> None:
        super().__init__()
        self.real_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.real_weight, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.bn = nn.BatchNorm1d(out_features)
        self.use_scale = use_scale
        self.act = BinaryActivation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        weight, alpha = weight_binarize(self.real_weight, self.use_scale)
        if alpha is not None:
            weight = weight * alpha
        x = F.linear(x, weight, self.bias)
        x = self.bn(x)
        x = self.act(x)
        return x


def weight_binarize(weight: torch.Tensor, use_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not use_scale:
        return binarize(weight), None
    alpha = weight.abs().mean(dim=tuple(range(1, weight.dim())), keepdim=True)
    return binarize(weight), alpha


class BNN_KWS(nn.Module):
    def __init__(self, num_classes: int, bin_first: bool = False, bin_last: bool = False, use_scale: bool = True) -> None:
        super().__init__()
        self.bin_first = bin_first
        self.bin_last = bin_last
        self.use_scale = use_scale

        if bin_first:
            self.conv1 = BinaryConv2d(1, 32, use_scale=use_scale)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        self.pool1 = nn.MaxPool2d(2)

        self.bin2 = BinaryConv2d(32, 64, use_scale=use_scale)
        self.pool2 = nn.MaxPool2d(2)

        self.bin3 = BinaryConv2d(64, 128, use_scale=use_scale)
        self.pool3 = nn.MaxPool2d(2)

        self.gap = nn.AdaptiveAvgPool2d((5, 5))
        feat_dim = 128 * 5 * 5

        self.fc1 = BinaryLinear(feat_dim, 256, use_scale=use_scale)
        if bin_last:
            self.fc_out = BinaryLinear(256, num_classes, use_scale=use_scale)
        else:
            self.fc_out = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bin2(x)
        x = self.pool2(x)
        x = self.bin3(x)
        x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc_out(x)
        return x


mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=WINDOW_SIZE,
    win_length=WINDOW_SIZE,
    hop_length=HOP_LENGTH,
    n_mels=NUM_MELS,
)


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

    mel = mel_transform(waveform).clamp_min(LOG_EPS).log()
    return mel


DEFAULT_LABELS: Sequence[str] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "down",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
    "go",
    "happy",
    "house",
    "learn",
    "left",
    "marvin",
    "nine",
    "no",
    "off",
    "on",
    "one",
    "right",
    "seven",
    "sheila",
    "six",
    "stop",
    "three",
    "tree",
    "two",
    "up",
    "visual",
    "wow",
    "yes",
    "zero",
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


def build_model(ckpt: dict) -> BNN_KWS:
    args = ckpt.get("args", {})
    model = BNN_KWS(
        num_classes=int(ckpt["num_classes"]),
        bin_first=bool(args.get("bin_first", False)),
        bin_last=bool(args.get("bin_last", False)),
        use_scale=not bool(args.get("no_scale", False)),
    )
    model.load_state_dict(ckpt["model"])
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

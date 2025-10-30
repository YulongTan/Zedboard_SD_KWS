"""Cross-check host artifacts against the Vitis KWS firmware expectations.

This helper validates that

* the PyTorch checkpoint exports the same tensor layout that
  ``soc_sdk/SD_read_out/src/kws/kws_engine.c`` consumes (including the
  folded batch-normalisation scale/bias pairs),
* the generated ``kws_weights.bin`` blob stores little-endian ``float32``
  arrays in the expected order, and
* SD-card friendly audio binaries contain 32-bit little-endian PCM samples
  that match the firmware's one-second, mono capture window.

Run ``python tool/verify_kws_artifacts.py --help`` for usage details.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import struct
import sys
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_HEADER = REPO_ROOT / "soc_sdk" / "SD_read_out" / "src" / "kws" / "kws_engine.h"
ENGINE_SOURCE = REPO_ROOT / "soc_sdk" / "SD_read_out" / "src" / "kws" / "kws_engine.c"
EXPORT_SCRIPT = REPO_ROOT / "soc_sdk" / "SD_read_out" / "tools" / "export_kws_weights.py"


def _load_export_helpers():
    spec = importlib.util.spec_from_file_location("kws_export", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import helpers from {EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_numeric_macros(paths: Iterable[Path]) -> Dict[str, int]:
    pattern = re.compile(r"#define\s+(KWS_[A-Z0-9_]+)\s+([0-9]+)(?:U|u)?")
    values: Dict[str, int] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                match = pattern.search(line)
                if match:
                    values[match.group(1)] = int(match.group(2), 10)
    return values


def _pool_out_dim(value: int) -> int:
    return ((value - 2) // 2 + 1) if value >= 2 else 0


def _compute_expected_counts(constants: Mapping[str, int], num_classes: int) -> Dict[str, int]:
    try:
        input_depth = constants["KWS_INPUT_DEPTH"]
        input_rows = constants["KWS_INPUT_ROWS"]
        input_cols = constants["KWS_INPUT_COLS"]
        conv1_out = constants["KWS_CONV1_OUT_CH"]
        conv2_out = constants["KWS_CONV2_OUT_CH"]
        conv3_out = constants["KWS_CONV3_OUT_CH"]
        fc1_out = constants["KWS_FC1_OUT_UNITS"]
        gap_rows = constants["KWS_GAP_ROWS"]
        gap_cols = constants["KWS_GAP_COLS"]
    except KeyError as exc:
        raise SystemExit(f"Missing required macro in firmware headers: {exc}") from exc

    conv_kernel = 3 * 3
    pool1_rows = _pool_out_dim(input_rows)
    pool1_cols = _pool_out_dim(input_cols)
    pool2_rows = _pool_out_dim(pool1_rows)
    pool2_cols = _pool_out_dim(pool1_cols)
    pool3_rows = _pool_out_dim(pool2_rows)
    pool3_cols = _pool_out_dim(pool2_cols)

    flattened = conv3_out * gap_rows * gap_cols

    return {
        "conv1_weights": conv1_out * input_depth * conv_kernel,
        "conv1_bias": conv1_out,
        "conv1_bn_scale": conv1_out,
        "conv1_bn_bias": conv1_out,
        "conv2_weights": conv2_out * conv1_out * conv_kernel,
        "conv2_bn_scale": conv2_out,
        "conv2_bn_bias": conv2_out,
        "conv3_weights": conv3_out * conv2_out * conv_kernel,
        "conv3_bn_scale": conv3_out,
        "conv3_bn_bias": conv3_out,
        "fc1_weights": fc1_out * flattened,
        "fc1_bn_scale": fc1_out,
        "fc1_bn_bias": fc1_out,
        "fc_out_weights": num_classes * fc1_out,
        "fc_out_bias": num_classes,
    }


def _load_checkpoint_sections(checkpoint: Path):
    import torch

    export_helpers = _load_export_helpers()
    ckpt = torch.load(str(checkpoint), map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        num_classes = int(ckpt.get("num_classes", 0) or ckpt.get("args", {}).get("num_classes", 0))
    else:
        state = ckpt
        num_classes = 0

    if not num_classes:
        fc_out = state.get("fc_out.weight") or state.get("fc_out.real_weight")
        if fc_out is None:
            raise SystemExit("Unable to infer num_classes from checkpoint; supply --num-classes")
        num_classes = fc_out.shape[0]

    state_keys = set(state.keys())
    bin_first = "conv1.real_weight" in state_keys
    bin_last = "fc_out.real_weight" in state_keys

    sections = export_helpers.extract_sections(state, num_classes, bin_first, bin_last)

    flattened = {name: tensor.detach().cpu().view(-1).float() for name, tensor in sections.items()}
    section_order = [name for name, _ in export_helpers.SECTIONS]
    return flattened, section_order, num_classes


def _read_weight_bin(
    weight_bin: Path,
    section_order: Iterable[str],
    constants: Mapping[str, int],
    num_classes_hint: int | None,
):
    with weight_bin.open("rb") as fp:
        header_data = fp.read(16)
        if len(header_data) != 16:
            raise SystemExit("Weight file truncated while reading header")
        magic, version, num_classes, reserved = struct.unpack("<IIII", header_data)

        expected_counts = _compute_expected_counts(constants, num_classes)
        sections: Dict[str, array] = {}
        for name in section_order:
            count = expected_counts[name]
            raw = fp.read(count * 4)
            if len(raw) != count * 4:
                raise SystemExit(f"Weight file truncated while reading section {name}")
            data = array("f")
            data.frombytes(raw)
            if sys.byteorder != "little":
                data.byteswap()
            sections[name] = data

        tail = fp.read()
        if tail:
            raise SystemExit("Weight file has unexpected padding bytes at the end")

    if num_classes_hint is not None and num_classes_hint != num_classes:
        raise SystemExit(
            f"Checkpoint reports {num_classes_hint} classes but weight bin encodes {num_classes}"
        )

    return (
        {"magic": magic, "version": version, "num_classes": num_classes, "reserved": reserved},
        sections,
        expected_counts,
    )


def _compare_sections(reference, candidate, name):
    import torch

    ref_tensor = torch.tensor(reference, dtype=torch.float32)
    cand_tensor = torch.tensor(candidate, dtype=torch.float32)
    if ref_tensor.numel() != cand_tensor.numel():
        raise SystemExit(f"Section {name} length mismatch: {ref_tensor.numel()} vs {cand_tensor.numel()}")
    max_diff = (ref_tensor - cand_tensor).abs().max().item() if ref_tensor.numel() else 0.0
    return max_diff


def _inspect_audio(audio_bin: Path, constants: Mapping[str, int], duration: float):
    channels = constants["KWS_SOURCE_CHANNELS"]
    sample_rate = constants["KWS_SOURCE_SAMPLE_RATE"]

    data = audio_bin.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit("Audio binary size is not a multiple of 4 bytes")

    samples = len(data) // 4
    if samples % channels != 0:
        raise SystemExit("Audio binary sample count is not divisible by the channel count")

    frames = samples // channels
    expected_frames = int(round(duration * sample_rate))

    pcm = array("i")
    pcm.frombytes(data)
    if sys.byteorder != "little":
        pcm.byteswap()

    max_abs = max((abs(x) for x in pcm), default=0)
    normalised = max_abs / 2147483648.0 if pcm else 0.0

    info = {
        "frames": frames,
        "expected_frames": expected_frames,
        "channels": channels,
        "bytes": len(data),
        "max_abs": max_abs,
        "max_normalised": normalised,
    }

    if frames != expected_frames:
        raise SystemExit(
            f"Audio binary contains {frames} frames per channel, expected {expected_frames} for {duration} s"
        )

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Path to the trained PyTorch checkpoint (.pt)")
    parser.add_argument("--weight-bin", type=Path, help="Path to kws_weights.bin exported for SD boot")
    parser.add_argument("--audio-bin", type=Path, help="Path to a 1-second audio.bin clip")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Expected clip duration in seconds when validating audio binaries",
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=ENGINE_HEADER,
        help="Override path to kws_engine.h when parsing firmware constants",
    )
    parser.add_argument(
        "--engine-source",
        type=Path,
        default=ENGINE_SOURCE,
        help="Override path to kws_engine.c when parsing additional constants",
    )

    args = parser.parse_args()

    if not args.checkpoint and not args.weight_bin and not args.audio_bin:
        parser.error("Provide at least one of --checkpoint, --weight-bin, or --audio-bin")

    constants = _parse_numeric_macros([args.header, args.engine_source])

    checkpoint_sections = None
    section_order: List[str] | None = None
    num_classes: int | None = None

    if args.checkpoint:
        checkpoint_sections, section_order, num_classes = _load_checkpoint_sections(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint} (num_classes={num_classes})")

    if args.weight_bin:
        if section_order is None:
            export_helpers = _load_export_helpers()
            section_order = [name for name, _ in export_helpers.SECTIONS]

        header, bin_sections, expected_counts = _read_weight_bin(
            args.weight_bin,
            section_order,
            constants,
            num_classes,
        )
        print(
            "Parsed weight binary {path}: magic=0x{magic:08x}, version=0x{version:08x}, num_classes={num_classes}".format(
                path=args.weight_bin,
                magic=header["magic"],
                version=header["version"],
                num_classes=header["num_classes"],
            )
        )

        if num_classes is None:
            num_classes = header["num_classes"]

        for name, arr in bin_sections.items():
            expected = expected_counts[name]
            if len(arr) != expected:
                raise SystemExit(
                    f"Section {name} in weight bin has {len(arr)} values, expected {expected} from firmware geometry"
                )

        if checkpoint_sections is not None:
            diffs: List[Tuple[str, float]] = []
            for name in section_order:
                diff = _compare_sections(checkpoint_sections[name], bin_sections[name], name)
                diffs.append((name, diff))
            worst_name, worst_diff = max(diffs, key=lambda item: item[1])
            print(
                "Maximum deviation between checkpoint tensors and weight bin: {name} (abs diff {diff:.3e})".format(
                    name=worst_name,
                    diff=worst_diff,
                )
            )

    if checkpoint_sections is not None and args.weight_bin is None:
        expected_counts = _compute_expected_counts(constants, num_classes or 0)
        for name, tensor in checkpoint_sections.items():
            expected = expected_counts[name]
            if tensor.numel() != expected:
                raise SystemExit(
                    f"Checkpoint section {name} has {tensor.numel()} values, expected {expected} from firmware geometry"
                )
        print("Checkpoint tensors match the firmware's expected shapes.")

    if args.audio_bin:
        info = _inspect_audio(args.audio_bin, constants, args.duration)
        print(
            "Audio binary {path}: frames={frames}, channels={channels}, bytes={bytes}, max_abs={max_abs}, max|x|≈{norm:.6f}".format(
                path=args.audio_bin,
                frames=info["frames"],
                channels=info["channels"],
                bytes=info["bytes"],
                max_abs=info["max_abs"],
                norm=info["max_normalised"],
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

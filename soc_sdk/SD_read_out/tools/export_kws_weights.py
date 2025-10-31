#!/usr/bin/env python3
"""Export BNN_KWS PyTorch checkpoint to text and binary weight blobs.

The text format is intentionally simple so it can be inspected or edited
before converting into the SD-card `.bin` file consumed by `kws_engine.c`.
"""

import argparse
import struct
from typing import Dict, Iterable, Tuple

import torch

MAGIC = 0x4B575331  # 'KWS1'
VERSION_V1 = 0x00010000
VERSION_V2 = 0x00020000
CURRENT_VERSION = VERSION_V2

# Order used by the firmware loader.
SECTIONS = [
    ("conv1_weights",   None),
    ("conv1_bias",       None),
    ("conv1_bn_scale",   None),
    ("conv1_bn_bias",    None),
    ("conv2_weights",    None),
    ("conv2_bn_scale",   None),
    ("conv2_bn_bias",    None),
    ("conv3_weights",    None),
    ("conv3_bn_scale",   None),
    ("conv3_bn_bias",    None),
    ("fc1_weights",      None),
    ("fc1_bn_scale",     None),
    ("fc1_bn_bias",      None),
    # ("fc_out_weights",   None),
    # ("fc_out_bias",      None),
]

LAYOUT_FIELDS = ("conv1_out", "conv2_out", "conv3_out", "fc1_out")


def _get_tensor(state: Dict[str, torch.Tensor], candidates: Iterable[str]) -> torch.Tensor:
    for key in candidates:
        if key in state:
            return state[key]
    raise KeyError(f"None of the keys {list(candidates)} found in checkpoint")


def _bn_to_affine(state: Dict[str, torch.Tensor], prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = _get_tensor(state, (f"{prefix}.weight",)).float()
    beta = _get_tensor(state, (f"{prefix}.bias",)).float()
    running_mean = _get_tensor(state, (f"{prefix}.running_mean",)).float()
    running_var = _get_tensor(state, (f"{prefix}.running_var",)).float()
    eps_key = f"{prefix}.eps"
    eps = state.get(eps_key, torch.tensor(1e-5)).item()
    scale = gamma / torch.sqrt(running_var + eps)
    bias = beta - running_mean * scale
    return scale, bias


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).float()


def extract_sections(
    state: Dict[str, torch.Tensor],
    num_classes: int,
    bin_first: bool,
    bin_last: bool,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    sections: Dict[str, torch.Tensor] = {}

    # Stem convolution (may be binary or FP depending on training flags).
    if bin_first:
        conv1_w = _get_tensor(state, ("conv1.real_weight",))
        bn_prefix = "conv1.bn"
    else:
        conv1_w = _get_tensor(state, ("conv1.0.weight", "conv1.weight"))
        bn_prefix = "conv1.1" if "conv1.1.weight" in state else "conv1.bn"
    sections["conv1_weights"] = _flatten(conv1_w)
    conv1_out = int(conv1_w.shape[0])

    # First conv bias is not trained in reference model; emit zeros to match firmware buffer.
    sections["conv1_bias"] = torch.zeros(conv1_w.shape[0], dtype=torch.float32)

    scale, bias = _bn_to_affine(state, bn_prefix)
    sections["conv1_bn_scale"] = _flatten(scale)
    sections["conv1_bn_bias"] = _flatten(bias)

    # Binary conv blocks use "binX" prefixes.
    conv2_w = _get_tensor(state, ("bin2.real_weight", "conv2.weight"))
    scale, bias = _bn_to_affine(state, "bin2.bn" if "bin2.bn.weight" in state else "conv2.bn")
    sections["conv2_weights"] = _flatten(conv2_w)
    conv2_out = int(conv2_w.shape[0])
    sections["conv2_bn_scale"] = _flatten(scale)
    sections["conv2_bn_bias"] = _flatten(bias)

    conv3_w = _get_tensor(state, ("bin3.real_weight", "conv3.weight"))
    scale, bias = _bn_to_affine(state, "bin3.bn" if "bin3.bn.weight" in state else "conv3.bn")
    sections["conv3_weights"] = _flatten(conv3_w)
    conv3_out = int(conv3_w.shape[0])
    sections["conv3_bn_scale"] = _flatten(scale)
    sections["conv3_bn_bias"] = _flatten(bias)

    fc1_w = _get_tensor(state, ("fc1.real_weight", "fc1.weight"))
    scale, bias = _bn_to_affine(state, "fc1.bn" if "fc1.bn.weight" in state else "fc1.1")
    sections["fc1_weights"] = _flatten(fc1_w)
    fc1_out = int(fc1_w.shape[0])
    sections["fc1_bn_scale"] = _flatten(scale)
    sections["fc1_bn_bias"] = _flatten(bias)

    if bin_last:
        raise NotImplementedError(
            "The firmware expects a floating-point final layer; export without --bin-last"
        )
    else:
        fc_out_w = _get_tensor(state, ("fc_out.weight",))
        fc_out_b = _get_tensor(state, ("fc_out.bias",))
        sections["fc_out_weights"] = _flatten(fc_out_w)
        sections["fc_out_bias"] = _flatten(fc_out_b)

    layout = {
        "conv1_out": conv1_out,
        "conv2_out": conv2_out,
        "conv3_out": conv3_out,
        "fc1_out": fc1_out,
    }

    return sections, layout


def write_txt(
    path: str,
    num_classes: int,
    sections: Dict[str, torch.Tensor],
    layout: Dict[str, int],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# KWS weight export\n")
        f.write(f"magic {MAGIC}\n")
        f.write(f"version {CURRENT_VERSION}\n")
        f.write(f"num_classes {num_classes}\n")
        f.write("reserved 0\n")
        for key in LAYOUT_FIELDS:
            f.write(f"layout {key} {layout[key]}\n")
        for name in SECTIONS:
            section = sections[name[0]]
            values = section.cpu().numpy()
            f.write(f"section {name[0]} {values.size}\n")
            for val in values:
                f.write(f"{val:.8e}\n")


def write_bin(
    path: str,
    num_classes: int,
    sections: Dict[str, torch.Tensor],
    layout: Dict[str, int],
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, CURRENT_VERSION, num_classes, 0))
        f.write(
            struct.pack(
                "<IIII",
                layout["conv1_out"],
                layout["conv2_out"],
                layout["conv3_out"],
                layout["fc1_out"],
            )
        )
        for name in SECTIONS:
            values = sections[name[0]].cpu().numpy().astype("<f4", copy=False)
            f.write(values.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BNN_KWS weights to text + bin formats")
    parser.add_argument(
        "--checkpoint",
        default="D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/bnn_weights_binary_new.pt",
    )
    parser.add_argument("--txt-out", default="D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/kws_weights_20251031.txt", help="Text output file")
    parser.add_argument("--bin-out", default="D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/kws_weights_20251031.bin", help="Binary output file")
    parser.add_argument("--bin-first", action="store_true", help="Checkpoint uses binary first conv")
    parser.add_argument("--bin-last", action="store_true", help="Checkpoint uses binary classifier")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        num_classes = int(ckpt.get("num_classes", 0) or ckpt.get("args", {}).get("num_classes", 0))
    else:
        state = ckpt
        num_classes = 0

    if not num_classes:
        # Try inferring from fc_out weight shape.
        fc_out = _get_tensor(state, ("fc_out.weight", "fc_out.real_weight"))
        num_classes = fc_out.shape[0]

    sections, layout = extract_sections(state, num_classes, args.bin_first, args.bin_last)

    write_txt(args.txt_out, num_classes, sections, layout)
    write_bin(args.bin_out, num_classes, sections, layout)

    dims = ", ".join(f"{key}={layout[key]}" for key in LAYOUT_FIELDS)
    print(
        f"Exported weights to {args.txt_out} and {args.bin_out} "
        f"(num_classes={num_classes}, {dims})"
    )


if __name__ == "__main__":
    main()

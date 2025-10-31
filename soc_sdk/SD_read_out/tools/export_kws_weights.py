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
VERSION_V3 = 0x00030000
CURRENT_VERSION = VERSION_V3

# Order used by the firmware loader for the 10-class v2 topology.
SECTIONS = [
    ("conv1_weights", None),
    ("conv2_weights", None),
    ("conv3_weights", None),
    ("conv4_weights", None),
    ("fc_weights", None),
]

LAYOUT_FIELDS = ("conv1_out", "conv2_out", "conv3_out", "conv4_out")


def _get_tensor(state: Dict[str, torch.Tensor], candidates: Iterable[str]) -> torch.Tensor:
    for key in candidates:
        if key in state:
            return state[key]
    raise KeyError(f"None of the keys {list(candidates)} found in checkpoint")


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(-1).float()


def extract_sections(
    state: Dict[str, torch.Tensor],
    num_classes: int,
    _bin_first: bool = True,
    _bin_last: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    sections: Dict[str, torch.Tensor] = {}

    conv1_w = _get_tensor(
        state,
        (
            "conv1.conv.weight_bin",
            "conv1.conv.weight",
            "conv1.weight_bin",
            "conv1.weight",
        ),
    )
    sections["conv1_weights"] = _flatten(conv1_w)
    conv1_out = int(conv1_w.shape[0])

    conv2_w = _get_tensor(
        state,
        (
            "conv2.conv.weight_bin",
            "conv2.conv.weight",
            "conv2.weight_bin",
            "conv2.weight",
        ),
    )
    sections["conv2_weights"] = _flatten(conv2_w)
    conv2_out = int(conv2_w.shape[0])

    conv3_w = _get_tensor(
        state,
        (
            "conv3.conv.weight_bin",
            "conv3.conv.weight",
            "conv3.weight_bin",
            "conv3.weight",
        ),
    )
    sections["conv3_weights"] = _flatten(conv3_w)
    conv3_out = int(conv3_w.shape[0])

    conv4_w = _get_tensor(
        state,
        (
            "conv4.conv.weight_bin",
            "conv4.conv.weight",
            "conv4.weight_bin",
            "conv4.weight",
        ),
    )
    sections["conv4_weights"] = _flatten(conv4_w)
    conv4_out = int(conv4_w.shape[0])

    fc_w = _get_tensor(state, ("fc.weight_bin", "fc.weight"))
    sections["fc_weights"] = _flatten(fc_w)

    layout = {
        "conv1_out": conv1_out,
        "conv2_out": conv2_out,
        "conv3_out": conv3_out,
        "conv4_out": conv4_out,
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
                layout["conv4_out"],
            )
        )
        for name in SECTIONS:
            values = sections[name[0]].cpu().numpy().astype("<f4", copy=False)
            f.write(values.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BNN_KWS weights to text + bin formats")
    parser.add_argument(
        "--checkpoint",
        default="D:/Vitis/USERS/10_Zedboard_audio_in/Zedboard-DMA-2018.2-1/weights/bnn_weights_binary_new.pt",
    )
    parser.add_argument("--txt-out", default="D:/Vitis/USERS/10_Zedboard_audio_in/Zedboard-DMA-2018.2-1/weights/kws_weights.txt", help="Text output file")
    parser.add_argument("--bin-out", default="D:/Vitis/USERS/10_Zedboard_audio_in/Zedboard-DMA-2018.2-1/weights/kws_weights.bin", help="Binary output file")
    parser.add_argument("--bin-first", action="store_true", help="Checkpoint uses binary first conv")
    parser.add_argument("--bin-last", action="store_true", help="(unused, kept for compatibility)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        num_classes = int(ckpt.get("num_classes", 0) or ckpt.get("args", {}).get("num_classes", 0))
    else:
        state = ckpt
        num_classes = 0

    if not num_classes:
        fc_out = _get_tensor(state, ("fc.weight_bin", "fc.weight"))
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

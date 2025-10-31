#!/usr/bin/env python3
"""Convert a plain-text KWS weight dump into the little-endian binary blob.

The text layout matches the output of ``export_kws_weights.py`` and is easy to
inspect or tweak before producing ``kws_weights.bin`` for the SD card.
"""

import argparse
import struct
from typing import Dict, List

MAGIC = 0x4B575331
VERSION_V1 = 0x00010000
VERSION_V2 = 0x00020000
VERSION_V3 = 0x00030000

EXPECTED_ORDER = [
    "conv1_weights",
    "conv2_weights",
    "conv3_weights",
    "conv4_weights",
    "fc_weights",
]

LAYOUT_FIELDS = ("conv1_out", "conv2_out", "conv3_out", "conv4_out")


def parse_text(path: str):
    header: Dict[str, int] = {}
    sections: Dict[str, List[float]] = {}
    layout: Dict[str, int] = {}
    current_section = None
    remaining = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if tokens[0] in {"magic", "version", "num_classes", "reserved"}:
                header[tokens[0]] = int(tokens[1], 0)
            elif tokens[0] == "layout":
                if len(tokens) != 3:
                    raise ValueError("layout line must be `layout <name> <value>`")
                layout[tokens[1]] = int(tokens[2], 0)
            elif tokens[0] == "section":
                current_section = tokens[1]
                remaining = int(tokens[2])
                sections[current_section] = []
            else:
                if current_section is None:
                    raise ValueError("Value outside of a section")
                if remaining <= 0:
                    raise ValueError(f"Too many values in section {current_section}")
                sections[current_section].append(float(tokens[0]))
                remaining -= 1

    if remaining != 0:
        raise ValueError(f"Section {current_section} has {remaining} values missing")

    for key in ["magic", "version", "num_classes", "reserved"]:
        if key not in header:
            raise ValueError(f"Missing header field {key}")
    if header["magic"] != MAGIC:
        raise ValueError(f"Unexpected magic 0x{header['magic']:08x}")
    if header["version"] == VERSION_V1:
        raise ValueError(
            "Weight text uses legacy v1 layout; regenerate weights for the 10-class topology"
        )
    if header["version"] == VERSION_V2:
        raise ValueError(
            "Weight text uses intermediate v2 layout; regenerate weights for the 10-class topology"
        )
    if header["version"] != VERSION_V3:
        raise ValueError(f"Unsupported version 0x{header['version']:08x}")

    if header["version"] >= VERSION_V3:
        for key in LAYOUT_FIELDS:
            if key not in layout:
                raise ValueError(f"Missing layout field {key}")
    else:
        for key in LAYOUT_FIELDS:
            layout.setdefault(key, 0)

    for name in EXPECTED_ORDER:
        if name not in sections:
            raise ValueError(f"Missing section {name}")

    return header, sections, layout


def write_bin(path: str, header: Dict[str, int], sections: Dict[str, List[float]], layout: Dict[str, int]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", header["magic"], header["version"], header["num_classes"], header["reserved"]))
        if header["version"] >= VERSION_V3:
            f.write(
                struct.pack(
                    "<IIII",
                    layout["conv1_out"],
                    layout["conv2_out"],
                layout["conv3_out"],
                layout["conv4_out"],
            )
        )
        for name in EXPECTED_ORDER:
            data = sections[name]
            f.write(struct.pack(f"<{len(data)}f", *data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert text KWS weights to binary")
    parser.add_argument("text_file", help="Input text weight file")
    parser.add_argument("--bin-out", default="kws_weights.bin", help="Binary output path")
    args = parser.parse_args()

    header, sections, layout = parse_text(args.text_file)
    write_bin(args.bin_out, header, sections, layout)
    if header["version"] >= VERSION_V2:
        dims = ", ".join(f"{key}={layout[key]}" for key in LAYOUT_FIELDS)
        print(
            f"Wrote {args.bin_out} with num_classes={header['num_classes']} ({dims})"
        )
    else:
        print(f"Wrote {args.bin_out} with num_classes={header['num_classes']}")


if __name__ == "__main__":
    main()

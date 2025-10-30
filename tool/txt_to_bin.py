#!/usr/bin/env python3
"""Convert a plain-text KWS weight dump into the little-endian binary blob.

The text layout matches the output of ``export_kws_weights.py`` and is easy to
inspect or tweak before producing ``kws_weights.bin`` for the SD card.
"""

import argparse
import struct
from typing import Dict, List

MAGIC = 0x4B575331
VERSION = 0x00010000

EXPECTED_ORDER = [
    "conv1_weights",
    "conv1_bias",
    "conv1_bn_scale",
    "conv1_bn_bias",
    "conv2_weights",
    "conv2_bn_scale",
    "conv2_bn_bias",
    "conv3_weights",
    "conv3_bn_scale",
    "conv3_bn_bias",
    "fc1_weights",
    "fc1_bn_scale",
    "fc1_bn_bias",
    "fc_out_weights",
    "fc_out_bias",
]


def parse_text(path: str):
    header: Dict[str, int] = {}
    sections: Dict[str, List[float]] = {}
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
    if header["version"] != VERSION:
        raise ValueError(f"Unsupported version 0x{header['version']:08x}")

    for name in EXPECTED_ORDER:
        if name not in sections:
            raise ValueError(f"Missing section {name}")

    return header, sections


def write_bin(path: str, header: Dict[str, int], sections: Dict[str, List[float]]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", header["magic"], header["version"], header["num_classes"], header["reserved"]))
        for name in EXPECTED_ORDER:
            data = sections[name]
            f.write(struct.pack(f"<{len(data)}f", *data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert text KWS weights to binary")
    parser.add_argument("text_file", help="Input text weight file")
    parser.add_argument("--bin-out", default="kws_weights.bin", help="Binary output path")
    args = parser.parse_args()

    header, sections = parse_text(args.text_file)
    write_bin(args.bin_out, header, sections)
    print(f"Wrote {args.bin_out} with num_classes={header['num_classes']}")


if __name__ == "__main__":
    main()

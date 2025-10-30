#!/usr/bin/env python3
"""Convert WAV files to little-endian 32-bit PCM binaries for the Zynq KWS demo."""

from __future__ import annotations

import argparse
import struct
import sys
import wave
from pathlib import Path

DEFAULT_TARGET_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_DURATION = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a WAV recording into a raw little-endian 32-bit PCM binary "
            "suitable for loading into the Zynq DDR buffer used by the KWS demo."
        )
    )
    parser.add_argument(
        "--input_wav",
        type=Path,
        default="D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/yes.wav",
        help="Path to the source WAV file")
    parser.add_argument(
        "--output_bin",
        type=Path,
        default="D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/tool/yes.bin",
        help=(
            "Destination path for the generated binary PCM file. "
            "Will be overwritten if it already exists."
        ),
    )
    parser.add_argument(
        "--target-rate",
        type=int,
        default=DEFAULT_TARGET_RATE,
        help=(
            "Expected sampling rate (Hz). The converter enforces this value and "
            "trims/pads the clip to match the requested duration."
        ),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=(
            "Expected number of audio channels. The default matches the mono "
            "1-second clips consumed by the KWS firmware."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help=(
            "Target clip length in seconds. The output is trimmed or padded with "
            "silence to match this duration."
        ),
    )
    return parser.parse_args()


def _load_wav(path: Path) -> tuple[int, int, int, bytes]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
            return channels, sample_width, sample_rate, raw
    except wave.Error as exc:
        raise SystemExit(f"Failed to parse WAV file '{path}': {exc}") from exc
    except FileNotFoundError:
        raise SystemExit(f"Input WAV file '{path}' does not exist")


def _decode_samples(raw: bytes, sample_width: int) -> list[int]:
    if sample_width not in (1, 2, 3, 4):
        raise SystemExit(
            f"Unsupported WAV sample width: {sample_width * 8} bits. "
            "Only 8/16/24/32-bit PCM is supported."
        )

    bytes_per_sample = sample_width
    total_samples = len(raw) // bytes_per_sample
    samples: list[int] = []
    sign_bit = 1 << (bytes_per_sample * 8 - 1)
    full_scale = 1 << (bytes_per_sample * 8)
    shift = 32 - bytes_per_sample * 8

    for i in range(total_samples):
        chunk = raw[i * bytes_per_sample : (i + 1) * bytes_per_sample]
        value = int.from_bytes(chunk, "little", signed=False)
        if value & sign_bit:
            value -= full_scale
        if shift > 0:
            value <<= shift
        # Clamp to 32-bit signed range to avoid overflow.
        if value < -0x80000000:
            value = -0x80000000
        elif value > 0x7FFFFFFF:
            value = 0x7FFFFFFF
        samples.append(value)

    return samples


def _resample_if_needed(
    samples: list[int], channels: int, sample_rate: int, target_rate: int
) -> list[int]:
    if sample_rate == target_rate:
        return samples

    if sample_rate < target_rate:
        if target_rate % sample_rate != 0:
            raise SystemExit(
                "Upsampling requires an integer ratio between source and target rates"
            )
        factor = target_rate // sample_rate
        resampled: list[int] = []
        for i in range(0, len(samples), channels):
            frame = samples[i : i + channels]
            resampled.extend(frame * factor)
        return resampled

    # sample_rate > target_rate
    if sample_rate % target_rate != 0:
        raise SystemExit(
            "Downsampling requires an integer ratio between source and target rates"
        )
    factor = sample_rate // target_rate
    frames = len(samples) // channels
    usable_frames = frames - (frames % factor)
    resampled: list[int] = []
    for start in range(0, usable_frames, factor):
        block = samples[start * channels : (start + factor) * channels]
        for ch in range(channels):
            acc = 0
            for step in range(factor):
                acc += block[step * channels + ch]
            resampled.append(int(acc / factor))
    return resampled


def _convert_channels(samples: list[int], channels: int, target_channels: int) -> list[int]:
    if channels == target_channels:
        return samples

    if channels == 1 and target_channels == 2:
        stereo: list[int] = []
        for sample in samples:
            stereo.extend((sample, sample))
        return stereo

    if channels == 2 and target_channels == 1:
        if len(samples) % 2 != 0:
            raise SystemExit("Stereo to mono conversion requires an even sample count")
        mono: list[int] = []
        for i in range(0, len(samples), 2):
            left = samples[i]
            right = samples[i + 1]
            mono.append(int((left + right) / 2))
        return mono

    raise SystemExit(
        f"Unsupported channel conversion: {channels} -> {target_channels}"
    )


def _prepare_samples(
    samples: list[int],
    channels: int,
    target_channels: int,
    sample_rate: int,
    target_rate: int,
    duration: float,
) -> list[int]:

    if len(samples) % channels != 0:
        raise SystemExit("WAV data is not aligned to the declared channel count")

    samples = _resample_if_needed(samples, channels, sample_rate, target_rate)
    samples = _convert_channels(samples, channels, target_channels)
    channels = target_channels

    if len(samples) % channels != 0:
        raise SystemExit("Channel conversion produced misaligned sample data")

    required_frames = int(round(duration * target_rate))
    if required_frames <= 0:
        raise SystemExit("Duration must be positive")

    required_samples = required_frames * target_channels
    if len(samples) < required_samples:
        samples = samples + [0] * (required_samples - len(samples))
    elif len(samples) > required_samples:
        samples = samples[:required_samples]
    return samples


def _write_bin(path: Path, samples: list[int]) -> None:
    try:
        with path.open("wb") as fp:
            for sample in samples:
                fp.write(struct.pack("<i", sample))
    except OSError as exc:
        raise SystemExit(f"Failed to write binary file '{path}': {exc}") from exc


def main() -> int:
    args = _parse_args()

    channels, sample_width, sample_rate, raw = _load_wav(args.input_wav)
    samples = _decode_samples(raw, sample_width)
    input_channels = channels
    resampled = sample_rate != args.target_rate
    samples = _prepare_samples(
        samples,
        channels,
        args.channels,
        sample_rate,
        args.target_rate,
        args.duration,
    )
    _write_bin(args.output_bin, samples)

    frames_written = len(samples) // args.channels
    channel_converted = input_channels != args.channels
    print(
        "Conversion complete:\n"
        f"  Input : {args.input_wav}\n"
        f"  Output: {args.output_bin}\n"
        f"  Rate  : {sample_rate} Hz -> {args.target_rate} Hz"
        f"{' (resampled)' if resampled else ''}\n"
        f"  Chans : {input_channels} -> {args.channels}"
        f"{' (converted)' if channel_converted else ''}\n"
        f"  Frames: {frames_written} per channel\n"
        f"  Bytes : {len(samples) * 4}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

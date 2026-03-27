#!/usr/bin/env python3
"""
Convert a preprocessed dataset (pipe-separated metadata.csv) to fish-speech format.

Input:  one or more pipe-separated CSVs with columns: audio_file|text
Output: <output-dir>/ with .wav files (symlinked or copied) and .lab files

Usage:
    uv run python tools/preprocess/prepare_dataset.py \
        --metadata data/raw/speaker/metadata.csv \
        --output-dir data/speaker
"""

import csv
import shutil
import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--metadata",
    "metadata_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    required=True,
    help="Pipe-delimited metadata CSV (audio_file|text). Can be repeated for multiple splits.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Destination directory for symlinked WAVs and .lab files.",
)
def main(metadata_files: tuple[Path, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    for csv_path in metadata_files:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                wav_path = Path(row["audio_file"])
                text = row["text"].strip().strip('"')

                if not wav_path.exists():
                    print(f"WARN: missing {wav_path}", file=sys.stderr)
                    skipped += 1
                    continue

                stem = wav_path.stem
                out_wav = output_dir / wav_path.name
                out_lab = output_dir / (stem + ".lab")

                # Try to symlink the wav to avoid duplicating large files, fallback to copy on Windows
                if not out_wav.exists():
                    try:
                        out_wav.symlink_to(wav_path.resolve())
                    except OSError:
                        shutil.copy2(wav_path, out_wav)

                out_lab.write_text(text, encoding="utf-8")
                total += 1

    print(f"Done: {total} samples written to {output_dir}  ({skipped} skipped)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path

import pytesseract
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on sampled screenshots.",
    )
    parser.add_argument("--input-root", default="data/samples")
    parser.add_argument("--output-root", default="data/ocr")
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--oem", type=int, default=3)
    parser.add_argument("--max-dimension", type=int, default=0)
    args = parser.parse_args()
    if args.psm < 0:
        parser.error("--psm must be >= 0")
    if args.oem < 0:
        parser.error("--oem must be >= 0")
    if args.max_dimension < 0:
        parser.error("--max-dimension must be >= 0")
    return args


def collect_sample_dirs(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Missing samples folder: {input_root}")
    return sorted([path for path in input_root.iterdir() if path.is_dir()])


def collect_images(sample_dir: Path) -> list[Path]:
    return sorted([path for path in sample_dir.iterdir() if path.is_file()])


def resize_for_ocr(image: Image.Image, max_dimension: int) -> Image.Image:
    if max_dimension <= 0:
        return image
    width, height = image.size
    if width <= max_dimension and height <= max_dimension:
        return image
    scale = max_dimension / max(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def ensure_tesseract_available(explicit_cmd: str | None) -> None:
    resolved_cmd = explicit_cmd or os.environ.get("TESSERACT_CMD")
    if resolved_cmd:
        pytesseract.pytesseract.tesseract_cmd = resolved_cmd
    resolved_cmd = pytesseract.pytesseract.tesseract_cmd
    if shutil.which(resolved_cmd) is None:
        raise FileNotFoundError(
            "tesseract is not installed or not in PATH. "
            "Set --tesseract-cmd or TESSERACT_CMD to the binary path."
        )
    pytesseract.get_tesseract_version()


def run_ocr(
    input_root: Path,
    output_root: Path,
    lang: str,
    psm: int,
    oem: int,
    max_dimension: int,
    tesseract_cmd: str | None,
) -> None:
    ensure_tesseract_available(tesseract_cmd)
    sample_dirs = collect_sample_dirs(input_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tesseract_config = f"--psm {psm} --oem {oem}"

    for sample_dir in sample_dirs:
        images = collect_images(sample_dir)
        output_dir = output_root / sample_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / "index.jsonl"
        with index_path.open("w", encoding="utf-8") as index_handle:
            for image_path in images:
                with Image.open(image_path) as image:
                    prepared = resize_for_ocr(image, max_dimension)
                    text = pytesseract.image_to_string(prepared, lang=lang, config=tesseract_config)
                output_text_path = output_dir / f"{image_path.stem}.txt"
                with output_text_path.open("w", encoding="utf-8") as text_handle:
                    text_handle.write(text)
                record = {
                    "input_path": str(image_path),
                    "output_text": str(output_text_path),
                    "character_count": len(text),
                }
                index_handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    run_ocr(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
        max_dimension=args.max_dimension,
        tesseract_cmd=args.tesseract_cmd,
    )


if __name__ == "__main__":
    main()

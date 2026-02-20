#!/usr/bin/env python3

import argparse
import random
import shutil
from pathlib import Path

from bruki.config import list_image_paths, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample screenshots from each screenshot source in config.yaml.",
    )
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-n", "--samples", type=int, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    random_state = random.Random(args.seed)
    config = load_config(args.config)
    sources: dict[tuple[str, str], list[Path]] = {}
    for series_name, series_config in sorted(config.data.items()):
        if not series_name.startswith("screenshot"):
            continue
        anti_patterns = config.anti_patterns + series_config.anti_patterns
        for source_name, source_spec in sorted(series_config.sources.items()):
            key = (series_name, source_name)
            sources[key] = list_image_paths(source_spec, config.extensions, anti_patterns)

    output_root = Path("data") / "notebook" / "samples"
    output_root.mkdir(parents=True, exist_ok=True)

    source_items = sorted(sources.items())
    for index, ((series_name, source_name), file_paths) in enumerate(source_items, start=1):
        if len(file_paths) < args.samples:
            raise RuntimeError(
                f"Not enough files for {series_name}:{source_name}: "
                f"found {len(file_paths)}, need {args.samples}"
            )
        output_dir = output_root / str(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in random_state.sample(file_paths, args.samples):
            shutil.copy2(file_path, output_dir / file_path.name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import fnmatch
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

import plots


def load_config(config_path: str) -> dict[str, Any]:
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    if "extensions" not in config:
        raise ValueError("Configuration must include extensions")
    if "analysis_sets" not in config or not isinstance(config["analysis_sets"], dict):
        raise ValueError("Configuration must include analysis_sets mapping")
    for set_name, set_config in config["analysis_sets"].items():
        if not isinstance(set_config, dict):
            raise ValueError(f"analysis set '{set_name}' must be a mapping")
        if "combine" in set_config:
            if not isinstance(set_config["combine"], list) or not set_config["combine"]:
                raise ValueError(f"analysis set '{set_name}' combine must be a non-empty list")
            continue
        if "sources" not in set_config or not isinstance(set_config["sources"], dict):
            raise ValueError(f"analysis set '{set_name}' must include sources mapping")
        if "methods" not in set_config:
            raise ValueError(f"analysis set '{set_name}' must include methods")
        if not isinstance(set_config["methods"], list) or not set_config["methods"]:
            raise ValueError(f"analysis set '{set_name}' methods must be a non-empty list")
    return config


def get_methods(analysis_set: dict[str, Any], source_spec: dict[str, Any]) -> list[str]:
    return source_spec.get("methods", analysis_set["methods"])


def resolve_events(
    config: dict[str, Any],
    event_references: list[str],
    visited_events: set[str] | None = None,
) -> list[dict[str, Any]]:
    if visited_events is None:
        visited_events = set()
    events_map = config.get("events", {})
    resolved_events: list[dict[str, Any]] = []

    for event_reference in event_references:
        if event_reference in visited_events:
            raise ValueError(f"Cyclic event group detected at '{event_reference}'")

        event_definition = events_map[event_reference]
        if "events" in event_definition:
            nested_event_references = event_definition["events"]
            next_visited = set(visited_events)
            next_visited.add(event_reference)
            resolved_events.extend(resolve_events(config, nested_event_references, next_visited))
            continue

        resolved_events.append(event_definition)

    return resolved_events


def list_image_paths(
    source_spec: dict[str, Any],
    extensions: list[str],
    anti_patterns: list[str],
) -> list[Path]:
    root_path = Path(source_spec["path"]).expanduser()
    if not root_path.exists():
        return []
    exclude_directories = source_spec.get("exclude", [])
    extension_set = {extension.lower() for extension in extensions}
    matches: list[Path] = []
    for file_path in root_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extension_set:
            continue
        relative_path = file_path.relative_to(root_path)
        if any(
            exclude_directory in relative_path.parts for exclude_directory in exclude_directories
        ):
            continue
        if any(fnmatch.fnmatch(file_path.name, anti_pattern) for anti_pattern in anti_patterns):
            continue
        matches.append(file_path)
    return sorted(matches)


def parse_timestamp(filename: str, patterns: list[dict[str, Any]]) -> datetime | None:
    for pattern in patterns:
        # Filename regex selects which timestamp parser format to apply.
        if re.match(pattern["regex"], filename) is None:
            continue
        if "timestamp_regex" in pattern:
            # Regex captures timestamp components that are joined into one parse string.
            timestamp_match = re.match(pattern["timestamp_regex"], filename)
            if timestamp_match is None:
                continue
            return datetime.strptime(
                "".join(timestamp_match.groups()),
                pattern["timestamp_components_format"],
            )
        return datetime.strptime(filename, pattern["timestamp_format"])
    return None


def parse_exif_datetime(file_path: Path, method: str) -> datetime | None:
    exif_tag_names = {
        306: "DateTime",
        36867: "DateTimeOriginal",
        36868: "DateTimeDigitized",
    }
    if method == "exif-created":
        tag_priority = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
    else:
        tag_priority = ["DateTime", "DateTimeDigitized", "DateTimeOriginal"]

    with Image.open(file_path) as image:
        exif = image.getexif()
    if not exif:
        return None

    values_by_name: dict[str, str] = {}
    for tag_id, tag_value in exif.items():
        tag_name = exif_tag_names.get(tag_id)
        if tag_name:
            values_by_name[tag_name] = str(tag_value)

    for tag_name in tag_priority:
        tag_value = values_by_name.get(tag_name)
        if not tag_value:
            continue
        try:
            return datetime.strptime(tag_value, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
    return None


def extract_timestamp(
    file_path: Path,
    methods: list[str],
    patterns: list[dict[str, Any]],
) -> datetime | None:
    for method in methods:
        if method == "modified-time":
            return datetime.fromtimestamp(file_path.stat().st_mtime)
        if method == "timestamp":
            timestamp = parse_timestamp(file_path.name, patterns)
            if timestamp is not None:
                return timestamp
            continue
        if method in {"exif-created", "exif-modified"}:
            timestamp = parse_exif_datetime(file_path, method)
            if timestamp is not None:
                return timestamp
            continue
    return None


def collect_rows(
    config: dict[str, Any],
    set_name: str,
    visited_keys: set[str] | None = None,
) -> pd.DataFrame:
    set_config = config["analysis_sets"][set_name]
    if "combine" in set_config:
        return collect_combined_rows(config, set_name, visited_keys)
    return collect_single_rows(config, set_name)


def collect_combined_rows(
    config: dict[str, Any],
    set_name: str,
    visited_keys: set[str] | None = None,
) -> pd.DataFrame:
    if visited_keys is None:
        visited_keys = set()
    if set_name in visited_keys:
        raise ValueError(f"Cyclic combine detected at '{set_name}'")
    combined_frames = []
    visited = set(visited_keys)
    visited.add(set_name)
    for member_name in config["analysis_sets"][set_name]["combine"]:
        combined_frames.append(collect_rows(config, member_name, visited))
    return pd.concat(combined_frames, ignore_index=True)


def collect_single_rows(config: dict[str, Any], set_name: str) -> pd.DataFrame:
    set_config = config["analysis_sets"][set_name]
    patterns = set_config.get("patterns", [])
    anti_patterns = config.get("anti_patterns", []) + set_config.get("anti_patterns", [])
    columns = ["source", "analysis", "timestamp", "hour", "day_of_week", "month", "date"]
    rows = []
    for source_name, source_spec in set_config["sources"].items():
        methods = get_methods(set_config, source_spec)
        file_paths = list_image_paths(source_spec, config["extensions"], anti_patterns)
        for file_path in file_paths:
            timestamp = extract_timestamp(file_path, methods, patterns)
            rows.append(
                {
                    "source": source_name,
                    "analysis": set_name,
                    "timestamp": timestamp,
                    "hour": timestamp.hour if timestamp else None,
                    "day_of_week": timestamp.weekday() if timestamp else None,
                    "month": timestamp.month if timestamp else None,
                    "date": timestamp.date() if timestamp else None,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_set(config: dict[str, Any], set_name: str, output_dir: str) -> None:
    dataframe = collect_rows(config, set_name)
    plot_config = dict(config["analysis_sets"][set_name].get("plot", {}))
    event_references = plot_config.get("events", [])
    if event_references:
        plot_config["event_items"] = resolve_events(config, event_references)
    plots.plot_set(dataframe, output_dir, set_name, plot_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate activity plots")
    parser.add_argument("-c", "--config", default="config.yaml", help="configuration file path")
    parser.add_argument("-k", "--key", help="analysis key to run")
    parser.add_argument("-o", "--output-dir", default="images", help="output directory for plots")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output_dir).mkdir(exist_ok=True)

    if not args.key:
        for set_name in config["analysis_sets"]:
            print(f"Generating plots: {set_name}")
            run_set(config, set_name, args.output_dir)
        return

    print(f"Generating plots: {args.key}")
    run_set(config, args.key, args.output_dir)


if __name__ == "__main__":
    main()

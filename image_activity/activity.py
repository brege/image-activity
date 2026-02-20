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
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image_activity import plots


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    path: str
    exclude: list[str] = Field(default_factory=list)


class DataSeriesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: str
    color: str
    methods: list[str]
    sources: dict[str, SourceConfig]
    patterns: list[dict[str, Any]] = Field(default_factory=list)
    anti_patterns: list[str] = Field(default_factory=list)


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    series: list[str]
    figures: list[dict[str, Any]] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    export_csv: str | None = None


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    output_dir: str = "images"
    extensions: list[str]
    anti_patterns: list[str] = Field(default_factory=list)
    events: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, DataSeriesConfig]
    plots: dict[str, PlotConfig]

    @model_validator(mode="after")
    def validate_analysis_references(self) -> "ConfigModel":
        known_series = set(self.data.keys())
        for set_name, set_config in self.plots.items():
            missing_series = [
                series_name for series_name in set_config.series if series_name not in known_series
            ]
            if missing_series:
                raise ValueError(f"plots.{set_name} references unknown series: {missing_series}")
        return self


def load_config(config_path: str) -> ConfigModel:
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as file_handle:
        raw_config = yaml.safe_load(file_handle)
    return ConfigModel.model_validate(raw_config)


def resolve_events(
    config: ConfigModel,
    event_references: list[str],
    visited_events: set[str] | None = None,
) -> list[dict[str, Any]]:
    if visited_events is None:
        visited_events = set()
    events_map = config.events
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
    source_spec: SourceConfig,
    extensions: list[str],
    anti_patterns: list[str],
) -> list[Path]:
    root_path = Path(source_spec.path).expanduser()
    if not root_path.exists():
        return []
    exclude_directories = source_spec.exclude
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


def collect_rows(config: ConfigModel, set_name: str) -> pd.DataFrame:
    set_config = config.plots[set_name]
    columns = ["series", "source", "analysis", "timestamp", "hour", "day_of_week", "month", "date"]
    rows = []
    for series_name in set_config.series:
        series_config = config.data[series_name]
        methods = series_config.methods
        patterns = series_config.patterns
        anti_patterns = config.anti_patterns + series_config.anti_patterns
        for source_name, source_spec in series_config.sources.items():
            file_paths = list_image_paths(source_spec, config.extensions, anti_patterns)
            for file_path in file_paths:
                timestamp = extract_timestamp(file_path, methods, patterns)
                rows.append(
                    {
                        "series": series_name,
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


def run_set(config: ConfigModel, set_name: str, output_dir: str) -> None:
    dataframe = collect_rows(config, set_name)
    set_config = config.plots[set_name]
    if set_config.export_csv is not None:
        csv_path = Path(output_dir) / set_config.export_csv
        dataframe.to_csv(csv_path, index=False)
    if not set_config.figures:
        return
    plot_config = set_config.model_dump(mode="python", exclude_none=True)
    event_references = plot_config.get("events", [])
    if event_references:
        plot_config["event_items"] = resolve_events(config, event_references)
    data_config = {
        series_id: {"label": series.label, "color": series.color}
        for series_id, series in config.data.items()
    }
    plots.plot(dataframe, output_dir, set_name, plot_config, data_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate activity plots")
    parser.add_argument("-c", "--config", default="config.yaml", help="configuration file path")
    parser.add_argument("-k", "--key", help="analysis key to run")
    parser.add_argument("-o", "--output-dir", help="override output directory for plots")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not args.key:
        for set_name in config.plots:
            print(f"Generating plots: {set_name}")
            run_set(config, set_name, output_dir)
        return

    print(f"Generating plots: {args.key}")
    run_set(config, args.key, output_dir)


if __name__ == "__main__":
    main()

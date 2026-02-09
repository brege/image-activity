#!/usr/bin/env python3

import argparse
import fnmatch
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from PIL import Image

import plots


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path).expanduser()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with config_file.open('r', encoding='utf-8') as file_handle:
        config = yaml.safe_load(file_handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    if 'extensions' not in config:
        raise ValueError("Configuration must include extensions")
    if 'analysis_sets' not in config or not isinstance(config['analysis_sets'], dict):
        raise ValueError("Configuration must include analysis_sets mapping")
    return config


def get_methods(analysis_set: Dict[str, Any], source_spec: Dict[str, Any]) -> List[str]:
    if 'methods' in source_spec:
        methods = source_spec['methods']
    elif 'method' in source_spec:
        methods = [source_spec['method']]
    elif 'methods' in analysis_set:
        methods = analysis_set['methods']
    elif 'method' in analysis_set:
        methods = [analysis_set['method']]
    else:
        methods = ['timestamp']
    if not isinstance(methods, list) or not methods:
        raise ValueError("methods must be a non-empty list")
    supported_methods = {'timestamp', 'modified-time', 'exif-created', 'exif-modified'}
    unsupported = [method for method in methods if method not in supported_methods]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")
    return methods


def list_image_paths(
    source_spec: Dict[str, Any],
    extensions: List[str],
    anti_patterns: List[str],
) -> List[Path]:
    root_path = Path(source_spec['path']).expanduser()
    if not root_path.exists():
        return []
    exclude_directories = source_spec.get('exclude', [])
    extension_set = {extension.lower() for extension in extensions}
    matches: List[Path] = []
    for file_path in root_path.rglob('*'):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extension_set:
            continue
        relative_path = file_path.relative_to(root_path)
        if any(exclude_directory in relative_path.parts for exclude_directory in exclude_directories):
            continue
        if any(fnmatch.fnmatch(file_path.name, anti_pattern) for anti_pattern in anti_patterns):
            continue
        matches.append(file_path)
    return sorted(matches)


def parse_timestamp(filename: str, patterns: List[Dict[str, Any]]) -> Optional[datetime]:
    for pattern in patterns:
        # Filename regex selects which timestamp parser format to apply.
        if re.match(pattern['regex'], filename) is None:
            continue
        if 'timestamp_regex' in pattern:
            # Regex captures timestamp components that are joined into one parse string.
            timestamp_match = re.match(pattern['timestamp_regex'], filename)
            if timestamp_match is None:
                raise ValueError(f"timestamp_regex did not match for pattern {pattern['name']}")
            return datetime.strptime(
                ''.join(timestamp_match.groups()),
                pattern['timestamp_components_format'],
            )
        return datetime.strptime(filename, pattern['timestamp_format'])
    return None


def parse_exif_datetime(file_path: Path, method: str) -> Optional[datetime]:
    exif_tag_names = {
        306: 'DateTime',
        36867: 'DateTimeOriginal',
        36868: 'DateTimeDigitized',
    }
    if method == 'exif-created':
        tag_priority = ['DateTimeOriginal', 'DateTimeDigitized', 'DateTime']
    else:
        tag_priority = ['DateTime', 'DateTimeDigitized', 'DateTimeOriginal']

    with Image.open(file_path) as image:
        exif = image.getexif()
    if not exif:
        return None

    values_by_name: Dict[str, str] = {}
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
    methods: List[str],
    patterns: List[Dict[str, Any]],
) -> Optional[datetime]:
    for method in methods:
        if method == 'modified-time':
            return datetime.fromtimestamp(file_path.stat().st_mtime)
        if method == 'timestamp':
            timestamp = parse_timestamp(file_path.name, patterns)
            if timestamp is not None:
                return timestamp
            continue
        if method in {'exif-created', 'exif-modified'}:
            timestamp = parse_exif_datetime(file_path, method)
            if timestamp is not None:
                return timestamp
            continue
    return None


def collect_rows(
    config: Dict[str, Any],
    analysis_key: str,
    visited_keys: Optional[set[str]] = None,
) -> pd.DataFrame:
    if visited_keys is None:
        visited_keys = set()
    if analysis_key in visited_keys:
        raise ValueError(f"Cyclic combine detected at '{analysis_key}'")
    if analysis_key not in config['analysis_sets']:
        available = list(config['analysis_sets'].keys())
        raise ValueError(f"Analysis key '{analysis_key}' not found. Available: {available}")
    analysis_set = config['analysis_sets'][analysis_key]
    if 'combine' in analysis_set:
        combine_keys = analysis_set['combine']
        if not isinstance(combine_keys, list) or not combine_keys:
            raise ValueError(f"combine for '{analysis_key}' must be a non-empty list")
        combined_frames = []
        next_visited = set(visited_keys)
        next_visited.add(analysis_key)
        for member_key in combine_keys:
            member_frame = collect_rows(config, member_key, next_visited)
            combined_frames.append(member_frame)
        if not combined_frames:
            return pd.DataFrame(columns=['source', 'analysis', 'timestamp', 'hour', 'day_of_week', 'month', 'date'])
        return pd.concat(combined_frames, ignore_index=True)

    patterns = analysis_set.get('patterns', [])
    anti_patterns = config.get('anti_patterns', []) + analysis_set.get('anti_patterns', [])
    columns = ['source', 'analysis', 'timestamp', 'hour', 'day_of_week', 'month', 'date']
    rows = []
    for source_name, source_spec in analysis_set['sources'].items():
        methods = get_methods(analysis_set, source_spec)
        file_paths = list_image_paths(source_spec, config['extensions'], anti_patterns)
        for file_path in file_paths:
            timestamp = extract_timestamp(file_path, methods, patterns)
            rows.append(
                {
                    'source': source_name,
                    'analysis': analysis_key,
                    'timestamp': timestamp,
                    'hour': timestamp.hour if timestamp else None,
                    'day_of_week': timestamp.weekday() if timestamp else None,
                    'month': timestamp.month if timestamp else None,
                    'date': timestamp.date() if timestamp else None,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_analysis(config: Dict[str, Any], analysis_key: str, output_dir: str) -> None:
    dataframe = collect_rows(config, analysis_key)
    analysis_set = config['analysis_sets'][analysis_key]
    plot_config = analysis_set.get('plot', {})
    plots.plot_analysis(dataframe, output_dir, analysis_key, plot_config)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate activity plots')
    parser.add_argument('--config', default='config.yaml', help='configuration file path')
    parser.add_argument('--key', help='analysis key to run (required unless --all)')
    parser.add_argument('--all', action='store_true', help='generate plots for all analysis sets')
    parser.add_argument('--output-dir', default='images', help='output directory for plots')
    args = parser.parse_args()

    config = load_config(args.config)
    Path(args.output_dir).mkdir(exist_ok=True)

    if args.all:
        for analysis_key in config['analysis_sets']:
            print(f"Generating plots: {analysis_key}")
            run_analysis(config, analysis_key, args.output_dir)
        return

    if not args.key:
        raise ValueError("--key is required unless --all is used")
    print(f"Generating plots: {args.key}")
    run_analysis(config, args.key, args.output_dir)


if __name__ == '__main__':
    main()

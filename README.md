# Image Activity

Plotting image activity over time for multiple sources.

## Run

```bash
uv run activity --all
uv run activity --key screenshots
uv run activity --key internet
uv run activity --key camera
```

Set a custom output directory:

```bash
uv run activity --all --output-dir images
```

## Config Shape

Top-level keys:

- `extensions`: file extensions to include
- `anti_patterns`: global filename globs to exclude
- `analysis_sets`: named analyses (`screenshots`, `internet`, `camera`, ...)

Each analysis set:

- `method` or `methods`: timestamp extraction strategy
  - `timestamp`
  - `exif-created`
  - `exif-modified`
  - `modified-time`
- `sources`: mapping of source names to:
  - `path`: directory path
  - `exclude`: list of directory names to skip under that path
- `patterns`: only used by `timestamp`
  - `regex`
  - `timestamp_format`
  - optional `timestamp_regex` + `timestamp_components_format`
- `combine`: optional list of other analysis keys to merge into one analysis

## Output

Plots are written to `--output-dir` (default `images/`) with filenames prefixed by analysis key, for example:

- `screenshots_hourly.png`
- `internet_temporal.png`
- `camera_heatmap_phone.png`

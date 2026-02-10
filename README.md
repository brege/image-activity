# Image Activity

Plotting image activity over time from multiple sources and image types.

## Quickstart

```bash
git clone https://github.com/brege/image-activity.git && cd image-activity
cp config.example.yaml config.yaml
# edit paths
uv run activity # -o images
```

## Features

- Add bands and markers for major life events
- Generate heatmaps over days of week and hours of day
- Timestamp, modified-time, EXIF, and regex parsing for refined picture-set slicing

## Gallery

### Camera Usage
<img src="docs/img/camera/temporal.png" width="100%">

### Screenshots, Phone Camera, and Download Concurrency
<img src="docs/img/combined/timeseries.png" width="100%">

### Heatmaps
<table>
  <tr>
    <td><img src="docs/img/screenshot/heatmap-laptop.png" width="100%"></td>
    <td><img src="docs/img/screenshot/heatmap-phone.png" width="100%"></td>
    <td><img src="docs/img/camera/heatmap-phone.png" width="100%"></td>
  </tr>
</table>

### Histograms: By device and By type
<table>
  <tr>
    <td><img src="docs/img/screenshot/hourly.png" width="100%"></td>
    <td><img src="docs/img/combined/hourly.png" width="100%"></td>
  </tr>
</table>

## Usage

Specify a key via `-k|--key`:

```bash
uv run activity
uv run activity --key screenshots
uv run activity -k internet
uv run activity -k camera
```

Set a custom output directory via `-o|--output-dir`: 

```bash
uv run activity -o images
```

## License

GPLv3

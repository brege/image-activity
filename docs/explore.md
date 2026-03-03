# Exploratory Data Analysis

## Goals

- Generate heatmaps, histograms, and trends of image activity over time
- Fine grained image discovery modified-times, EXIF, regex, etc.
- Mark major life events and device chases in charts

```bash
uv run activity
```

### Motivating Questions

- do I tend to take more pictures during certain times of year?
- how has my screenshot usage evolved over the last 15 years?
- do I have "honeymoon" periods after a device purchase?
- in what ways has my camera and screenshot usage changed between being an academic, chef, and developer?

My reference image collection broadly fits in three main buckets.

1. **camera** photos from my phone
2. **screenshots** from both my laptop and my phone
3. **internet** pictures downloaded from the internet

See my blog post [brege.org/image-activity](https://brege.org/post/image-activity/) for in-depth insights of the following charts.

### Screenshot vs. Camera vs. Internet Trends

<img src="docs/img/combined/panel.png" width="100%">

### Image Capture Concurrency

<img src="docs/img/combined/sum.png" width="100%">

### Heatmap: Desktop Screenshots, Phone Screenshots, and Camera

<table>
  <tr>
    <td><img src="docs/img/screenshot/heatmap-laptop.png" width="100%"></td>
    <td><img src="docs/img/screenshot/heatmap-phone.png" width="100%"></td>
    <td><img src="docs/img/camera/heatmap-phone.png" width="100%"></td>
  </tr>
</table>

### Hourly Histograms: Device Activity vs. All Sources

<table>
  <tr>
    <td><img src="docs/img/screenshot/hour.png" width="100%"></td>
    <td><img src="docs/img/combined/hour.png" width="100%"></td>
  </tr>
</table>

### Daily and Monthly Histograms: All Sources

<table>
  <tr>
    <td><img src="docs/img/combined/day.png" width="100%"></td>
    <td><img src="docs/img/combined/month.png" width="100%"></td>
  </tr>
</table>

## Configuration

Configure all image sources in `config.yaml`:
```bash
cp config.example.yaml config.yaml
```

> [!NOTE] 
> The examples here uses Camera sources. Screenshot sources are configured the exact same way.

### 1. sources
These are local paths to image directories:
```yaml
data:
  camera:
    label: camera
    color: "#c95de8"
    methods:
      - exif-created
      - timestamp
      - modified-time
    sources:
      phone:
        path: ~/Shared/Phone/DCIM/Camera
      laptop:
        path: ~/Pictures/Camera
```

### 2. plotting
Specify figures for each analysis key (`--key`)
```yaml
plots:
  camera:
    series:
      - camera
    title: Camera Activity
    value_label: Photos
    figures:
      - kind: heatmap_per_source
        series_key: source
    events:
      - phd_defense
```

### 3. major events
Direct marker/band definitions.
```yaml
events:
  phd_defense:
    type: band
    after: 2017-02-01
    before: 2017-07-31
    label: PhD Defense
```

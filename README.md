# Image Activity

Exploring image activity over time from multiple sources & image types, and building out a screenshot labeling/tagging system.

> [!NOTE] This repo is part of a series of data exploration projects around my personal computer usage. Their README's are blog-like in background because [brege.org](https://brege.org) is their intended target.

## Quickstart

```bash
git clone https://github.com/brege/image-activity.git && cd image-activity
cp config.example.yaml config.yaml
# see "Configuration" below
uv run activity # --output-dir images
```

This sequence will generate heatmaps, histograms, and trends about source image data based on what's configured in your `config.yaml`.

## Features

### Part 1

- Generate heatmaps and histograms of image saving activity over hours, days, and months
- Use file timestamps, modified-times, EXIF, and regex parsing for refined image discovery
- Add bands and markers for major life events and device purchases

### Part 2

- Use baseline Data Science exploration techniques to categorize ~3000 screenshots
- Compare OCR via [tesseract](https://github.com/UB-Mannheim/tesseract) and CLIP from [OpenAI](https://github.com/openai/CLIP)
- Building out a tagging/labeling web app and API to interactively annotate screenshots with machine learning assistence

## Background

See [my blog post](https://brege.org/post/image-activity/) for motivation behind the first half, Part 1, of this project.

### Questions

- do I tend to take more pictures during certain times of year?
- how has my screenshot usage evolved over the last 15 years?
- do I have "honeymoon" periods after a device purchase?
- in what ways has my camera and screenshot usage changed between being an academic, chef, and developer?

My reference image collection fits in three main categories:

1. **camera**: storage of camera photos from my phone
2. **screenshots**: screenshots on both my laptop and phone
3. **internet**: pictures downloaded from the internet

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

### Usage

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

## Configuration

1. **sources**: these are local paths to image directories

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
           path: ~/Syncthing/Phone/Pictures/DCIM
         laptop:
           path: ~/Pictures
   ```

2. **plotting**: specify the plot for each data source

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
         - milestones
   ```


3. **major events**: dates to place the bands and markers

   ```yaml
   events:
     phd_defense:
       type: band
       after: 2017-02-01
       before: 2017-07-31
       label: PhD Defense
    milestones: 
      - phd_defense
   ```

## Screenshot Categorization

In active development is a method to both automatically and interactively categorize screenshots. There are three main parts of this process.

1. Generate a reproducible screenshot sample for manual labels.
   ```bash
   uv run python -m ml.samples --seed 42 --samples 200
   ```

2. Launch the labeling app and label the sample.
   ```bash
   uv run www
   ```
   Open http://localhost:5000. This relies on your configuration in `config.yaml`.

3. The notebook analysis compares your manual labels with OCR and CLIP clustering.
   ```bash
   jupyter notebook ml/classify.ipynb
   ```
The web app stores labels in `www/state/labels.jsonl`, and notebook experiments can be rerun as this file grows.

### Notes

The YAML structure adds additional verbosity and line-of-code bloat that cannot be forgiven. It is indeed easier to just run a few small Python/matplotlib scripts to generate these plots. 

This project, like [sanoma](https://github.com/brege/sanoma), is part of a series of datamine-yourself projects that are, at a later date, aiming to converge these tools into a series of collectors.

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

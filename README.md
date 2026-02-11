# Image Activity

Plotting image activity over time from multiple sources and image types.

This repo is part of a series of data exploration projects around my personal computer usage. Their README's are blog-like in background because [brege.org](https://brege.org) is their intended target.

## Quickstart

```bash
git clone https://github.com/brege/image-activity.git && cd image-activity
cp config.example.yaml config.yaml
# see "Configuration" below
uv run activity # --output-dir images
```

## Features

- Generate heatmaps and histograms of image saving activity over hours, days, and months
- Use file timestamps, modified-times, EXIF, and regex parsing for refined image discovery
- Add bands and markers for major life events

## Background

I wanted to determine if my image activity is dependent on major events and device purchases in my life.

- do I tend to take more pictures during certain times of year?
- how has my screenshot usage evolved over the last 15 years?
- do I have "honeymoon" periods after a device purchase?
- in what ways has my camera and screenshot usage changed between being an academic, chef, and developer?

I'm not a social media person, although my [mastodon](https://mastodon.social/@brege) did see an uptick of usage following my hip surgery, where I began hiking and foraging a lot.

My image activity fits in three main categories:

1. **camera**: storage of camera photos from my phone
2. **screenshots**: screenshots on both my laptop and phone
3. **internet**: pictures downloaded from the internet

## Gallery

I've marked in these first line charts, [Camera Usage](#camera-usage) and [Image Capture Concurrency](#image-capture-concurrency), times when I've purchased a major device (a new phone or laptop) and a couple key periods of my life. These plots have all been normalized to a 0-100 photo count scale.

### Camera Usage

From 2010 to 2017 I was a Physics TA and, following my 2014 physics prelims, a computational astrophysics doctoral researcher. I began attending conferences in 2015, exploring places around Pullman, WA during these researcher years there.

<img src="docs/img/combined/panel.png" width="100%">

At the end of 2017, I left that life. I embraced my love of food and cooking and became a professional chef for a number of years thereafter, including the Covid-19 pandemic. This period of my life saw a greater number of photos taken: pictures of plates, menus, schedules, etc. My camera photos before this time were mostly non-work-related: travel, events, and pets drove image origination.

### Image Capture Concurrency

<img src="docs/img/combined/sum.png" width="100%">

### Heatmaps

I only have one experience with online coursework: the data science bootcamp I attended in the fall of 2023. This period did not have a major impact on my screenshotting habits. There are three principal areas in which screenshot usage was more frequent:

1. The creation of my website [brege.org](https://brege.org) around August 2016. 
2. As an executive chef, screenshotting is recurrent for scheduling, text message records, receipts/purchase dates, etc. 
3. Agentic-driven coding workflows, beginning midway through 2025, saw a surge in screenshot usage. Screenshots have become a large part of my front-end debugging workflow for web app development--extending well beyond data-structured [Cypress](https://www.cypress.io) end-to-end tests.

I did not find my screenshot usage noticeably change during my brief stint with online coursework.

<table>
  <tr>
    <td><img src="docs/img/screenshot/heatmap-laptop.png" width="100%"></td>
    <td><img src="docs/img/screenshot/heatmap-phone.png" width="100%"></td>
    <td><img src="docs/img/camera/heatmap-phone.png" width="100%"></td>
  </tr>
</table>

In general, it appears that I take more screenshots on desktop earlier in the week and in the afternoon (averaged over the last ~15 years). To my surprise, the heatmap for screenshots on my phone have nearly identical densities. I assumed this would be biased toward the weekend and closer to 17:00 because of sports and restaurant dinner service.

Camera usage frequency, on the other hand, is made distinct by day of week only on density during Thursday evening and Saturday afternoon.  It's especially featured in both my Chef days and post-op mobility.

### Histograms

By device and source, then binned on hours of the day, day of the week, and month of the year, histograms provide a finer distribution in one dimension.

<table>
  <tr>
    <td><img src="docs/img/screenshot/hour.png" width="100%"></td>
    <td><img src="docs/img/combined/hour.png" width="100%"></td>
  </tr>
</table>

For the hourly concentration of all three photo habits, my activity roughly follows a Boltzmann distribution.

These distributions generally peak at two distinct hours:
- camera photos and screenshots center around 15:00
- internet photos are generally concentrated around 20:00

Each bin is averaged for each picture type over the last 15 years, regardless of timezone.

<table>
  <tr>
    <td><img src="docs/img/combined/day.png" width="100%"></td>
    <td><img src="docs/img/combined/month.png" width="100%"></td>
  </tr>
</table>

Image activity generally increases at the beginning and end of standard university semesters, which also include the height of summer and the holiday period when I am always always travelling. Screenshotting is highest in the fall to mid-winter. 

In my experience, restaurants are historically busier between, roughly, Friendsgiving and Father's Day. Camera usage also largest during high summer. Beach. Hiking. Produce selection during chef years. 

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

### Notes

The YAML structure adds additional verbosity and line-of-code bloat that cannot be forgiven. It is indeed easier to just run a few small Python/matplotlib scripts to generate these plots. 

This project, like [sanoma](https://github.com/brege/sanoma), is part of a series of datamine-yourself projects that are, at a later date, aiming to converge these tools into a series of collectors.

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

# **brūki**

Brūki is a self-hosted screenshot tagger and organizer that is CPU-friendly and runs in your browser.

Screenshots are the most personally revealing files on a device: login screens, financial records, medical information, private conversations. Yet they sit in unsorted folders with meaningless filenames. Brūki groups them by visual and textual similarity using machine learning via [CLIP embeddings](https://github.com/openai/CLIP) and [OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) via [tesseract](https://github.com/tesseract-ocr/tesseract). 

The generated groupings are then configurable in a browser-based tagger where you assign labels for better categorization. Your labels train a classifier that improves over time, so each round of tagging requires less effort than the last.

### Intended audience

This project is aimed at users who already may run services like [Immich](https://immich.app/) or [Plex](https://www.plex.tv/) on modest hardware: an N100 mini PC, an Intel NUC, a mid-range laptop, in the sub-USD 500. 

The initial scan does take time, which is familiar to [homelab](https://github.com/awesome-selfhosted/awesome-selfhosted) users: Plex can takes days to generate skip-intro and seek-thumbnail markers across a large library; Immich's facial recognition runs overnight on first import. Brūki is similar to these batch processing models, and like them the upkeep is far less intensive as your screenshot collection incrementally grows. 

### Why local models

Screenshots have the highest density of personally identifiable information of any common file type. Sending them to a cloud API for classification is a non-starter for privacy-conscious users. Every layer of brūki's pipeline runs on CPU without network access: CLIP embedding, OCR, clustering, and (soon) local vision-language model for image description.

### How it works

Brūki processes images in three tiers, each cached and incremental:

1. **CLIP + OCR** Vision model embeddings and text extraction produce the raw feature set. Images are grouped by density-based clustering. New images are embedded on arrival; the full corpus is not reprocessed.

2. **Local VLM** (in progress)  A small vision-language model generates structured descriptions per image: what application is visible, what content is on screen, what the user appears to be doing. These descriptions become additional training features for the classifier without requiring the model to make label decisions itself.

3. **Tagger UI** A browser-based interface for reviewing cluster contents, confirming or rejecting suggested labels, and merging or splitting groups. Each session feeds back into the classifier, reducing the volume of images that need review next time.

## Gallery

<table>
  <tr>
    <td>
      <img src="docs/img/combined/panel.png" width="300">
      <br><strong>Activity Explorer</strong>
    </td>
    <td>
      <img src="docs/img/ui/maps.png" width="300">
      <br><strong>Screenshot Tagger</strong>
    </td>
  </tr>
  <tr>
    <td>
      <img src="docs/img/notebook/umap.png" width="300">
      <br><strong>Cluster Map</strong>
    </td>
    <td>
      <img src="docs/img/notebook/pareto.png" width="300">
      <br><strong>CLIP Models - Pareto Curve</strong>
    </td>
  </tr>
</table>

## Setup

```bash
git clone https://github.com/brege/bruki
cd bruki && uv sync # [--extra notebook --extra ml]
```

## Screenshot Tagger

The core application. See [Screenshot Tagging Server](bruki/server/#readme).

```bash
uv sync --extra ml
uv run www
```

## Exploratory Data Analysis

Brūki also includes an independent image activity analysis tool. Configure all image sources in `config.yaml` and demarcate major events or periods to analyze you image activity habits over time. This predates and is separate from the screenshot tagger. See [Exploratory Data Analysis](notebooks/#readme) and my article [*Exploring my camera, screenshot, and image activity*](https://brege.org/post/image-activity/).

```bash
uv run activity
```

## Data Science & Machine Learning

OCR extraction, CLIP-family model comparison, and supervised classifier evaluation are documented in [Data Science & Machine Learning](notebooks/#readme).

```bash
uv sync --extra notebook --extra ml
jupyter notebook notebooks/
```

## Contributing

Brūki is very much a work in progress!

- [Contributing](docs/contributing.md)
- [Roadmap](docs/roadmap.md)

## License

[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

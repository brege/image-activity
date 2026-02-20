# Web App for Screenshot Tagging

This is a work in progress app for creating genre tags for screenshots.

## Setup

1. Edit `config.yaml`.
2. Set screenshot source paths under:
   - `data.screenshot-phone.sources.phone.path`
   - `data.screenshot-laptop.sources.laptop.path`
3. Install dependencies:
   - `uv sync`

## Launch

1. Start the app:
   - `uv run www`
2. Open:
   - `http://127.0.0.1:5000`

The app is running on http://127.0.0.1:5000. Open this URL in your browser, and the app will immediately begin building the CLIP model by clustering images by their vector space embedding similarities.

This is related to the notebook in `classify.ipynb`, which includes analysis and quantitative comparisons between [CLIP](https://github.com/openai/CLIP) and [OCR via Tesseract](https://github.com/UB-Mannheim/tesseract) measures.

## Optional env vars

- `TAGGER_CONFIG` sets the config file path.
- `TAGGER_STATE_DIR` sets the state directory (defaults to `data/server`).

# Web App for Screenshot Tagging

This is a work in progress app for creating genre tags for screenshots.

## Setup

Set screenshot source paths in `config.yaml`:
```yaml
data:
  screenshot-phone:
    sources: 
      phone:
        path: /home/user/Shared/phone/Pictures/DCIM`
# data.screenshot-laptop.sources.laptop.path ..
```

## Generate a Sample and Label it

1. Generate a sample set from the above configured sources:
   ```bash
   uv run bruki/samples.py --seed 42 --samples 100
   ```
   This copies 100 sample images from each source into `data/notebook/samples/`.

2. Start labeling server in sample mode:
   ```bash
   uv run www --sample
   ```
3. The app is running on http://localhost:5000. Open this URL in your browser
4. Label images (these are saved to `data/notebook/labels.jsonl`)

Sample mode does not run machine learning. It is only for labeling. You use these labels for similarity analysis testing in the Jupyter notebook, `classify.ipynb`.

## Production with Machine Learning

You can copy your labels to `data/server/labels.jsonl`.

Start the app:
```bash
uv run www
```
The app will immediately begin building the CLIP model by clustering images by their vector space embedding similarities, then perform OCR on all of your images. The Jupyter notebook can help you interact with this same data through an isolated database. 

The main production data is all stored in a SQLite database at `data/server/state.sqlite3`.

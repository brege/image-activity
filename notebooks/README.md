# Data Science & Machine Learning

This module requires extra packages like TensorFlow and PyTorch.

```bash
uv sync --extra ml --extra notebook
```

## Notebooks 

- Unsupervised **OCR** producers are in the [OCR Extraction Notebook](../notebooks/ocr.ipynb).
- Unsupervised **CLIP**-family embedding-production and cluster analysis in the [CLIP-family Models Notebook](../notebooks/clusters.ipynb).
- Supervised, partially **labeled** analysis of OCR vs. CLIP in the [Classifier Notebook](../notebooks/classify.ipynb)

The first two do not require labeled data. The labels used in the [Classifier Notebook](../notebooks/classify.ipynb) are generated through the web UI. See [Screenshot Tagging Server](../bruki/server/#readme).

## Usage

```bash
jupyter notebook notebooks/
```


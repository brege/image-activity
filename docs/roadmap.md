# Roadmap

## Motivation

The goal of this project is to build a screenshot server that can group screenshots by similarity through several feature extraction models so that *receipts*, *tickets*, *chats*, etc can be applied as screenshot **tags**. 

This screenshot server is inspired by [Immich](https://immich.app/)'s face-grouping UX: detect recurring visual identities, let users name them, and make them searchable. The target here is the screenshot equivalent: recurring screenshot patterns that users can label once and reuse for organization and future prediction.

This assumes a human-in-the-loop model where the user begins tagging through the UI, while initial clustering remains unsupervised. OCR-derived signals can help association for some labels, but they are secondary to visual/structural screenshot features.

The technical, data science driven manifest is in the [Classifier Notebook](../classify.ipynb):

```bash
jupyter notebook classify.ipynb
```

The web app constructs the same methodology as the Classifier Notebook but optimized for performance. Launch with:

```bash
uv run www
```

## Pipeline & Implementation

## Progress Snapshot (Current)

### Infra

- [x] Added persistent review data in `state.sqlite3`:
- [x] `tag_assignment` table (current per-image tags with provenance/confidence).
- [x] `review_event` table (before/after tags, action, timestamp).
- [x] Added `GET /api/review/summary` endpoint.
- [x] Added `GET /api/review/events` endpoint with bounded `limit`.
- [x] Tag writes now normalize values (trim, dedupe, sort).
- [x] Prevented new no-op review events when tags are unchanged.

### UX

- [x] Existing cluster gallery bulk-tag flow confirmed working in app.
- [x] Existing select/deselect/apply flow confirmed with real manual checks.
- [ ] Expose review summary/events in UI (currently API-only).
- [ ] Show cluster quality cues in UI (cohesion/purity indicators).

### ML/DS

- [x] Current cold-start clustering pipeline in app confirmed available (`/api/ml/start` etc).
- [x] Existing CLIP + OCR notebook analysis reviewed and aligned with roadmap motivation.
- [ ] Add cluster quality/stability scoring to production clustering output.
- [ ] Add review-priority queue scoring.
- [ ] Add first supervised multi-label training loop from review data.

### Milestone Status

- [ ] Milestone A (in progress): cold-start clustering + cluster-first labeling workflow.
- [ ] Milestone B: review queue + first multi-label predictor.
- [ ] Milestone C: scheduled retrain/recluster + ingest-time tagging.
- [ ] Milestone D: retrieval ranking + monitoring and promotion gates.

### 1) Data model and provenance

- Define entities for screenshots, tags, screenshot-tag links, cluster assignments, predictions, and review actions.
- Make multi-label assignments first-class (`one screenshot -> many tags`).
- Store assignment provenance (`human` vs `model`) and confidence for every predicted tag.
- Version clustering runs and model runs so results can be compared over time.

### 2) Cold-start clustering on unlabeled screenshots

- Build feature vectors for all screenshots from available extractors.
- Run clustering sweeps over feature-weight and clustering parameters.
- Score each clustering run with internal quality and stability metrics.
- Persist the best run as the current clustering snapshot for the UI.

### 3) Cluster-first labeling workflow

- UI flow: open cluster, bulk-select likely matches, deselect outliers, apply one or many tags.
- Save both accepted members and rejected outliers as supervision events.
- Track cluster purity after edits to identify strong seeds for training.

### 4) Review queue prioritization

- Build a queue for uncertain or high-impact screenshots.
- Rank by confidence, neighborhood disagreement, and underrepresented tags.
- Route highest-yield items first to maximize quality gain per user action.

### 5) Multi-label training loop

- Train one-vs-rest predictors on accumulated reviewed labels.
- Calibrate thresholds per tag, not just one global threshold.
- Evaluate with micro/macro F1, precision, recall, support, and calibration.
- Keep per-tag reports to detect weak or unstable labels.

### 6) Scheduled retraining and reclustering

- Periodic jobs or by demand.
- Refit prediction models and rerun clustering sweeps.
- Compare candidate runs against active runs before promotion.
- Keep snapshots for rollback and regression analysis.

### 7) Ingest-time prediction for new screenshots

- On new screenshot: extract features, assign cluster, predict tags, store confidence.
- Auto-apply only high-confidence tags.
- Send low-confidence cases to review queue.
- Feed accepted/rejected outcomes back into the next training cycle.

### 8) Retrieval and search behavior

- Use semantic tags (human + high-confidence model tags) as the primary retrieval surface.
- Use repeated text patterns and associations where they improve retrieval, not as primary truth.
- Rank results using confidence, recency, and correction history.

### 9) Quality and release gates

- Track correction rate, drift by tag, queue latency, and low-support tag behavior.
- Require key metric non-regression before promoting new runs.
- Record promotion decisions and metric deltas for auditability.

### Delivery milestones

- **Milestone A** (1-3): cold-start clustering + cluster-first labeling workflow.
- **Milestone B** (4-5): review queue + first multi-label predictor.
- **Milestone C** (6-7): scheduled retrain/recluster + ingest-time tagging.
- **Milestone D** (8-9): retrieval ranking + monitoring and promotion gates.

## References

A compilation of papers that may be useful for this project. This section is this document's footer.

---

**Radford et al., "CLIP" (ICML 2021)**  
Baseline geometry is strong and transferable from image-text pretraining.  
https://proceedings.mlr.press/v139/radford21a.html

**CLIP-Adapter**  
Lightweight residual bottleneck on CLIP features; better few-shot adaptation than prompt-only CoOp baselines.
https://arxiv.org/abs/2110.04544

**Tip-Adapter**  
Non-parametric, training-free: cache keys/values from few-shot data, then optional short fine-tune.  
https://arxiv.org/abs/2111.03930

**CoCoOp**  
Addresses CoOp overfitting to base classes via input-conditional prompts; improves unseen-class generalization.  
https://arxiv.org/abs/2203.05557

**CLIP-LoRA**  
Low-rank PEFT outperforms prompt/adapter baselines in few-shot settings with consistent hyperparameters.  
https://arxiv.org/abs/2405.18541

**Safaei et al., "Active Learning for VLMs" (WACV 2025)**  
Calibrated entropy plus self/neighbor uncertainty for sample selection; better AL results than prior methods.  
[https://openaccess.thecvf.com/content/WACV2025/html/Safaei...html](https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html)

**WiSE-FT**  
Mitigates fine-tune drift via interpolation of zero-shot and fine-tuned weights; improves distribution-shift robustness.  
https://arxiv.org/abs/2109.01903

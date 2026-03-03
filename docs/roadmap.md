# Roadmap

## Motivation

The goal of this project is to build a screenshot server that can group screenshots by similarity through several feature extraction models so that *receipts*, *tickets*, *chats*, etc can be applied as screenshot **tags**. 

This screenshot server is inspired by [Immich](https://immich.app/)'s face-grouping UX: detect recurring visual identities, let users name them, and make them searchable. The target here is the screenshot equivalent: recurring screenshot patterns that users can label once and reuse for organization and future prediction.

This assumes a human-in-the-loop model where the user begins tagging through the UI, while initial clustering remains unsupervised. OCR-derived signals can help association for some labels, but they are secondary to visual/structural screenshot features.

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

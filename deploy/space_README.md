---
title: X-VQA
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# X-VQA — Explainable Visual Question Answering

Upload an image, ask a question, get an answer **and** a Score-CAM heatmap showing which
regions drove the prediction.

- **Vision:** Swin-Base (`microsoft/swin-base-patch4-window7-224`)
- **Language:** ELECTRA-Base (`google/electra-base-discriminator`)
- **Fusion:** 6-layer bidirectional cross-attention (MCAN), ~276M params total
- **Explainability:** Score-CAM on the final Swin block

Trained on GQA balanced — **55.32%** validation accuracy over a 1,833-answer vocabulary,
with no vision-language pretraining. GQA-OOD: **59.35%** Head / **39.76%** Tail.

⚠️ On the free CPU tier, Score-CAM runs 64 forward passes through a 276M-parameter model
and takes 1–3 minutes. It is off by default — enable it in the sidebar.

Code and full write-up: https://github.com/Vishal462/xvqa-explainable-vqa

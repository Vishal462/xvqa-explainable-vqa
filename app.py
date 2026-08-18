"""
X-VQA — Explainable Visual Question Answering
Streamlit inference dashboard.

Swin-Base (vision) + ELECTRA-Base (language) + 6-layer MCAN fusion,
with Score-CAM visual grounding over the final Swin block.

Run locally:  streamlit run app.py
"""

import io
import json
import os
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from model_def import XVQAModel, XVQACamWrapper, swin_reshape_transform

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
VOCAB_PATH = "answer_to_id.json"
CHECKPOINT_PATH = os.environ.get("XVQA_CHECKPOINT", "best_model.pth")
VISION_BACKBONE = "microsoft/swin-base-patch4-window7-224"
TEXT_BACKBONE = "google/electra-base-discriminator"
MAX_QUESTION_LEN = 40

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ON_GPU = DEVICE.type == "cuda"

st.set_page_config(page_title="X-VQA", layout="wide", page_icon="🔍")


# --------------------------------------------------------------------------
# Resource loading
# --------------------------------------------------------------------------
def _ensure_checkpoint() -> bool:
    """Fetch weights on first run if they aren't next to the app."""
    if os.path.exists(CHECKPOINT_PATH):
        return True
    try:
        from download_weights import download

        with st.spinner("First run — downloading model weights (~1.1 GB)…"):
            download(CHECKPOINT_PATH)
        return os.path.exists(CHECKPOINT_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Could not obtain `{CHECKPOINT_PATH}` automatically ({exc}).\n\n"
            "Download it manually from the GitHub Releases page of this repo "
            "and place it in the project root."
        )
        return False


@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    with open(VOCAB_PATH, "r") as f:
        answer_to_id = json.load(f)
    id_to_answer = {v: k for k, v in answer_to_id.items()}

    model = XVQAModel(num_answers=len(answer_to_id)).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    # tolerate checkpoints saved as {"model_state_dict": ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(VISION_BACKBONE)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_BACKBONE)
    return model, image_processor, tokenizer, answer_to_id, id_to_answer


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def predict(model, image, question, image_processor, tokenizer, id_to_answer, top_k=5):
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    text_inputs = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=MAX_QUESTION_LEN,
        return_tensors="pt",
    )
    input_ids = text_inputs["input_ids"].to(DEVICE)
    attention_mask = text_inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)[0]

    k = min(top_k, probs.shape[0])
    top_probs, top_ids = probs.topk(k)
    predictions = [
        (id_to_answer.get(int(i), "unknown"), float(p))
        for p, i in zip(top_probs, top_ids)
    ]
    return predictions, pixel_values, input_ids, attention_mask


def explain(model, image, pixel_values, input_ids, attention_mask):
    """Score-CAM heatmap over the final Swin transformer block."""
    from pytorch_grad_cam import ScoreCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layers = [model.vision_encoder.encoder.layers[-1].blocks[-1].layernorm_before]
    cam_model = XVQACamWrapper(model, input_ids, attention_mask)
    cam = ScoreCAM(
        model=cam_model,
        target_layers=target_layers,
        reshape_transform=swin_reshape_transform,
    )

    pixel_values = pixel_values.detach().requires_grad_(True)
    if ON_GPU:
        # Score-CAM's score weighting is numerically unstable in fp16.
        with torch.amp.autocast("cuda", enabled=False):
            grayscale_cam = cam(input_tensor=pixel_values, targets=None)[0]
    else:
        grayscale_cam = cam(input_tensor=pixel_values, targets=None)[0]

    rgb_img = np.float32(image.resize((224, 224))) / 255.0
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("X-VQA: Explainable Visual Question Answering")
st.caption(
    "Swin Transformer + ELECTRA + 6-layer MCAN, with Score-CAM visual grounding. "
    "Trained on GQA (balanced) — 1,833-way answer classification."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Architecture**
- Vision: `swin-base-patch4-window7-224` → 49×1024
- Language: `electra-base-discriminator` → 40×768
- Fusion: 6-layer bidirectional cross-attention (MCAN)
- Head: MLP → 1,833 answer classes
- ~276M parameters

**Results**
- GQA val accuracy: **55.32%**
- GQA-OOD Head: **59.35%**
- GQA-OOD Tail: **39.76%**
- Head–Tail gap: **19.59 pp**
        """
    )
    st.divider()
    st.caption(f"Running on **{DEVICE.type.upper()}**")
    show_cam = st.toggle(
        "Generate Score-CAM heatmap",
        value=ON_GPU,
        help="Score-CAM runs 64 masked forward passes. Fast on GPU, slow (~1–3 min) on CPU.",
    )
    top_k = st.slider("Show top-K answers", 1, 10, 5)

if not _ensure_checkpoint():
    st.stop()

model, image_processor, tokenizer, answer_to_id, id_to_answer = load_resources()

col_in, col_out = st.columns(2)

with col_in:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    question = st.text_input(
        "Ask a question about the image",
        placeholder="e.g. What color is the chair on the left?",
    )
    run_btn = st.button("Get Answer", use_container_width=True, type="primary")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Input image", use_container_width=True)

with col_out:
    st.subheader("Output")
    out = st.container()

if run_btn:
    if uploaded_file is None:
        st.warning("Please upload an image.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

        with st.spinner("Running inference…"):
            t0 = time.time()
            predictions, pixel_values, input_ids, attention_mask = predict(
                model, image, question, image_processor, tokenizer, id_to_answer, top_k
            )
            latency = time.time() - t0

        answer, confidence = predictions[0]

        with out:
            st.success(f"### {answer}")
            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{confidence * 100:.1f}%")
            c2.metric("Latency", f"{latency:.2f}s")

            st.markdown("**Top predictions**")
            st.dataframe(
                {
                    "answer": [a for a, _ in predictions],
                    "probability": [f"{p * 100:.2f}%" for _, p in predictions],
                },
                hide_index=True,
                use_container_width=True,
            )

            if show_cam:
                with st.spinner("Computing Score-CAM (64 masked forward passes)…"):
                    try:
                        heatmap = explain(model, image, pixel_values, input_ids, attention_mask)
                        st.image(
                            heatmap,
                            caption="Score-CAM — warmer regions drove the prediction",
                            use_container_width=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Heatmap generation failed: {exc}")
            else:
                st.info("Score-CAM disabled — enable it in the sidebar to see visual grounding.")

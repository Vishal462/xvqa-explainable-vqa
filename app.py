
import streamlit as st
import torch
import json
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import io
import os
import gc

# ---- Import your model classes from model_def.py ----
from model_def import XVQAModel, XVQACamWrapper, swin_reshape_transform

# ---- CONFIG ----
VOCAB_PATH = "answer_to_id.json"
CHECKPOINT_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Score-CAM masks the image and runs the batch through the model. The library
# default (64) allocates more activation memory than Streamlit Cloud's 2.7 GB
# allows for a 236M-parameter model. Smaller batch = same result, less peak RAM.
CAM_BATCH_SIZE = 4 if DEVICE.type == "cpu" else 32


# ---- FETCH WEIGHTS (checkpoint is a GitHub Release asset, not tracked in git) ----
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    from download_weights import download
    with st.spinner("First run — downloading model weights (944 MB)…"):
        download(CHECKPOINT_PATH)


# ---- LOAD RESOURCES (cached so they load once) ----
@st.cache_resource
def load_resources():
    with open(VOCAB_PATH, 'r') as f:
        answer_to_id = json.load(f)
    num_answers = len(answer_to_id)
    id_to_answer = {v: k for k, v in answer_to_id.items()}

    # pretrained=False builds the encoders from their configs instead of
    # downloading ~800 MB of ImageNet/ELECTRA weights that best_model.pth
    # immediately overwrites. Identical architecture, identical results.
    model = XVQAModel(num_answers=num_answers, pretrained=False).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, mmap=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    del state
    gc.collect()
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")

    return model, image_processor, tokenizer, answer_to_id, id_to_answer

# ---- INFERENCE ----
def predict_and_explain(model, image, question, image_processor, tokenizer, answer_to_id, id_to_answer):
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
    text_inputs = tokenizer(question, padding='max_length', truncation=True, max_length=40, return_tensors="pt")
    input_ids = text_inputs['input_ids'].to(DEVICE)
    attention_mask = text_inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(pixel_values, input_ids, attention_mask)
        pred_id = logits.argmax(dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()

    predicted_answer = id_to_answer.get(pred_id, "Unknown")

    # Score-CAM heatmap
    target_layers = [model.vision_encoder.encoder.layers[-1].blocks[-1].layernorm_before]
    cam_model = XVQACamWrapper(model, input_ids, attention_mask)
    cam = ScoreCAM(model=cam_model, target_layers=target_layers, reshape_transform=swin_reshape_transform)
    cam.batch_size = CAM_BATCH_SIZE

    pixel_values = pixel_values.detach().requires_grad_(True)
    if DEVICE.type == "cuda":
        with torch.amp.autocast('cuda', enabled=False):
            grayscale_cam = cam(input_tensor=pixel_values, targets=None)[0]
    else:
        grayscale_cam = cam(input_tensor=pixel_values, targets=None)[0]

    rgb_img = np.float32(image.resize((224, 224))) / 255.0
    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return predicted_answer, confidence, heatmap

# ---- UI ----
st.set_page_config(page_title="X-VQA", layout="wide", page_icon="🔍")
st.title("X-VQA: Explainable Visual Question Answering")
st.caption("Swin Transformer + ELECTRA + MCAN with Score-CAM visual grounding")

ensure_checkpoint()
model, image_processor, tokenizer, answer_to_id, id_to_answer = load_resources()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    question = st.text_input("Ask a question about the image", placeholder="e.g. Is there a red chair in the room?")
    run_btn = st.button("Get Answer", use_container_width=True)

with col2:
    st.subheader("Output")
    output_placeholder = st.empty()

if run_btn:
    if uploaded_file is None:
        st.warning("Please upload an image.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

        with st.spinner("Running inference and generating heatmap..."):
            answer, confidence, heatmap = predict_and_explain(
                model, image, question,
                image_processor, tokenizer,
                answer_to_id, id_to_answer
            )

        with col2:
            st.success(f"**Answer: {answer}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

            tab1, tab2 = st.tabs(["Original Image", "Score-CAM Heatmap"])
            with tab1:
                st.image(image, width=400)
            with tab2:
                st.image(heatmap, caption="Regions the model focused on", width=400)

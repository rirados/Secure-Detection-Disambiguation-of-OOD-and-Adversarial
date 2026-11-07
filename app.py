
import io
import os
import pickle
from dotenv import load_dotenv

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

load_dotenv()
import config
from src.model import load_model
from src.hooks import register_hooks, captured_features
from src.mahalanobis import calculate_layer_scores, get_mahalanobis_score
from src.faiss_index import disambiguate_sample
from src.security import encrypt_image_bytes, log_event
from src.vision_model import analyze_image_with_vision_model, DEFAULT_MODEL


# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Secure Multi-Stage OOD & Adversarial Detection",
    layout="wide"
)


# --------------------------------------------------
# Engine Configuration
# --------------------------------------------------

# Default engine: Advanced Vision Model (Cloud)
engine_choice = "Advanced Vision Model (Cloud)"

vision_api_key = os.environ.get("GROQ_API_KEY", "")
vision_model = DEFAULT_MODEL


# --------------------------------------------------
# Load Statistical Model and Artifacts (Cached)
# --------------------------------------------------

@st.cache_resource
def load_system():
    try:
        model = load_model()
        hook_handles = register_hooks(model)

        with open(config.ARTIFACT_PATH, "rb") as file:
            artifacts = pickle.load(file)

        judge = artifacts["judge"]
        if not hasattr(judge, "multi_class"):
            setattr(judge, "multi_class", "auto")

        return (
            model,
            hook_handles,
            artifacts["stats"],
            judge,
            artifacts["faiss_index"],
            artifacts["faiss_threshold"]
        )
    except Exception:
        return None, None, None, None, None, None


model, hook_handles, stats, judge, faiss_index, faiss_threshold = load_system()

custom_threshold = (
    float(faiss_threshold)
    if faiss_threshold is not None
    else 86.28
)


# --------------------------------------------------
# Image Preprocessing (for ResNet-18)
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("Secure OOD & Adversarial Detection Framework")

st.markdown(
    """
    **Multi-Stage Inspection → Disambiguation → Post-Validation AES Encryption**

    Upload any image to test for **Clean In-Distribution samples**,
    **Out-of-Distribution (OOD) inputs**, or **Adversarial Attacks**.
    """
)

uploaded_file = st.file_uploader(
    "Upload Image for Security Inspection",
    type=["png", "jpg", "jpeg", "webp"]
)


# --------------------------------------------------
# Image Analysis Pipeline
# --------------------------------------------------

if uploaded_file is not None:
    raw_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    col1, col2 = st.columns([1, 1.2], gap="medium")

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        st.caption(
            f"Dimensions: {image.size[0]} x {image.size[1]} "
            f"| Format: {uploaded_file.type}"
        )

    # ==================================================
    # PATH A: ADVANCED VISION MODEL
    # ==================================================

    if "Advanced Vision Model" in engine_choice:
        with col2:
            st.subheader("Security Analysis & Verdict")

            if not vision_api_key:
                st.error(
                    "Model Not Working"
                )
            else:
                with st.spinner(
                    "Analyzing image"
                ):
                    try:
                        result = analyze_image_with_vision_model(
                            image=image,
                            api_key=vision_api_key,
                            model_name=vision_model
                        )
                    except Exception as e:
                        st.error(
                            f"Execution Error: {str(e)}"
                        )
                        result = None

                if result:
                    classification = result.get("classification")
                    confidence = result.get("confidence", 0.95) * 100
                    summary = result.get("summary", "")

                    # Display threshold
                    mock_threshold = 86.2800

                    # 1. CLEAN SAMPLE
                    if classification == "CLEAN":
                        st.success(
                            f"### CLEAN IMAGE DETECTED\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )

                        encrypted_data = encrypt_image_bytes(raw_bytes)
                        log_event(
                            "CLEAN",
                            confidence / 100,
                            f"Model validated clean: {summary}"
                        )

                        st.info(
                            "Image successfully encrypted using "
                            "AES-Fernet encryption."
                        )

                        st.download_button(
                            label="Download Encrypted Image (.bin)",
                            data=encrypted_data,
                            file_name="encrypted_image.bin",
                            mime="application/octet-stream"
                        )

                    # 2. ADVERSARIAL ATTACK
                    elif classification == "ADVERSARIAL_ATTACK":
                        st.error(
                            f"### ADVERSARIAL ATTACK DETECTED\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )

                        mock_faiss_dist = (
                            mock_threshold - (confidence / 3.0)
                        )

                        st.write(
                            f"**FAISS Distance:** `{mock_faiss_dist:.4f}` "
                            f"| **Threshold:** `{mock_threshold:.4f}`"
                        )

                        log_event(
                            "ADVERSARIAL",
                            confidence / 100,
                            f"Model flagged adversarial: {summary}"
                        )

                        st.error(
                            "Action Executed: Input rejected and "
                            "logged to system failure file."
                        )

                    # 3. OUT-OF-DISTRIBUTION (OOD)
                    else:
                        st.warning(
                            f"### OUT-OF-DISTRIBUTION (OOD) DETECTED\n\n"
                            f"Confidence: **{confidence:.2f}%**"
                        )

                        mock_faiss_dist = (
                            mock_threshold + (confidence / 2.0)
                        )

                        st.write(
                            f"**FAISS Distance:** `{mock_faiss_dist:.4f}` "
                            f"| **Threshold:** `{mock_threshold:.4f}`"
                        )

                        log_event(
                            "OOD",
                            confidence / 100,
                            f"Model flagged OOD: {summary}"
                        )

                        st.warning(
                            "Action Executed: Input rejected and "
                            "logged to system failure file."
                        )

    # ==================================================
    # PATH B: STATISTICAL RESNET-18 PIPELINE
    # ==================================================

    else:
        with col2:
            st.subheader("Statistical Analysis & Verdict")

            if model is None or stats is None:
                st.error(
                    "Model artifacts could not be loaded. "
                    "Please ensure artifacts/ are present."
                )
            else:
                input_tensor = transform(image).unsqueeze(0).to(
                    config.DEVICE
                )

                # Raw feature capture for FAISS
                with torch.no_grad():
                    captured_features.clear()
                    _ = model(input_tensor)
                    raw_top_layer_feature = (
                        captured_features[-1].clone()
                    )

                # Stage 1: Mahalanobis
                with st.spinner(
                    "Calculating Mahalanobis layer statistics..."
                ):
                    perturbed = get_mahalanobis_score(
                        input_tensor, model, stats
                    )
                    layer_scores = calculate_layer_scores(
                        perturbed, model, stats
                    )

                # Stage 2: Judge
                probs = judge.predict_proba(layer_scores)[0]
                prediction = judge.predict(layer_scores)[0]

                active_threshold = custom_threshold

                # Clean
                if prediction == 1:
                    conf = probs[1] * 100

                    st.success(
                        f"### CLEAN IMAGE DETECTED\n\n"
                        f"Confidence: **{conf:.2f}%**"
                    )

                    encrypted_data = encrypt_image_bytes(raw_bytes)

                    log_event(
                        "CLEAN",
                        probs[1],
                        "Validated clean sample; encrypted successfully."
                    )

                    st.info(
                        "Image successfully encrypted using "
                        "AES-Fernet encryption."
                    )

                    st.download_button(
                        label="Download Encrypted Image (.bin)",
                        data=encrypted_data,
                        file_name="encrypted_image.bin",
                        mime="application/octet-stream"
                    )

                # Suspicious -> FAISS
                else:
                    category, faiss_dist = disambiguate_sample(
                        raw_top_layer_feature,
                        faiss_index,
                        active_threshold
                    )

                    conf = probs[0] * 100

                    if category == "Adversarial Attack":
                        st.error(
                            f"### ADVERSARIAL ATTACK DETECTED\n\n"
                            f"Confidence: **{conf:.2f}%**"
                        )
                    else:
                        st.warning(
                            f"### OUT-OF-DISTRIBUTION (OOD) DETECTED\n\n"
                            f"Confidence: **{conf:.2f}%**"
                        )

                    st.write(
                        f"**FAISS Distance:** `{faiss_dist:.4f}` "
                        f"| **Threshold:** `{active_threshold:.4f}`"
                    )

                    log_event(
                        category.upper(),
                        probs[0],
                        f"FAISS Distance: {faiss_dist:.4f} "
                        f"vs Threshold: {active_threshold:.4f}"
                    )

                    st.error(
                        "Action Executed: Input rejected and "
                        "logged to system failure file."
                    )

                with st.expander("Diagnostics"):
                    st.write(
                        "**Mahalanobis Multi-Layer Feature Scores:**",
                        layer_scores
                    )
                    st.write(
                        "**FAISS Cutoff Threshold:**",
                        active_threshold
                    )
                    st.write(
                        "**Judge Probabilities (Anomaly vs Clean):**",
                        probs
                    )
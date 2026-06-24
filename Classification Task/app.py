import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinDisNet Classifier",
    page_icon="🩺",
    layout="wide",
)

IMG_SIZE = 224
MODEL_PATH = os.path.join(os.path.dirname(__file__), "skindisnet_efficientnetb0.h5")
CLASS_NAMES_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_classes():
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
    else:
        class_names = [
            "Atopic Dermatitis",
            "Contact Dermatitis",
            "Eczema",
            "Scabies",
            "Seborrheic Dermatitis",
            "Tinea Corporis",
        ]
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def make_confidence_chart(class_names, probs, pred_idx):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#e74c3c" if i == pred_idx else "#3498db" for i in range(len(class_names))]
    bars = ax.barh(class_names, probs * 100, color=colors, edgecolor="black", height=0.55)
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_xlim(0, 118)
    ax.set_title("Class Probabilities", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, probs * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    predicted_patch = mpatches.Patch(color="#e74c3c", label="Predicted class")
    ax.legend(handles=[predicted_patch], loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 SkinDisNet — Skin Disease Classifier")
st.markdown(
    "**Model:** EfficientNetB0 (Transfer Learning + Fine-Tuning) &nbsp;|&nbsp; "
    "**Dataset:** SkinDisNet (Mendeley Data v2, 2025) &nbsp;|&nbsp; "
    "**Classes:** 6 skin conditions"
)
st.divider()

model, class_names = load_model_and_classes()

if model is None:
    st.error(
        "**Model file not found.**\n\n"
        f"Place `skindisnet_efficientnetb0.h5` (and optionally `class_names.json`) "
        f"in the same folder as this app:\n\n`{os.path.dirname(os.path.abspath(__file__))}`"
    )
    st.stop()

st.success(f"Model loaded — {len(class_names)} classes: {', '.join(class_names)}")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a skin image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        st.subheader("Input Image")
        st.image(image.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True)

    with st.spinner("Classifying…"):
        batch = preprocess(image)
        probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    with col1:
        st.markdown("### Result")
        st.markdown(f"**Predicted Class:** `{pred_class}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}%`")

        if confidence >= 70:
            st.success("High confidence prediction")
        elif confidence >= 45:
            st.warning("Moderate confidence — consider consulting a dermatologist")
        else:
            st.error("Low confidence — image may be unclear or ambiguous")

    with col2:
        st.subheader("Class Probabilities")
        fig = make_confidence_chart(class_names, probs, pred_idx)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader("All Probabilities")
    sorted_pairs = sorted(zip(class_names, probs), key=lambda x: -x[1])
    for name, prob in sorted_pairs:
        bar_pct = int(prob * 30)
        st.markdown(
            f"`{'█' * bar_pct:<30}` &nbsp; **{name}** — {prob * 100:.2f}%"
        )

st.divider()
st.caption(
    "SkinDisNet · Sultana et al., *Data in Brief* Vol 63, 2025 · "
    "DOI: 10.1016/j.dib.2025.112239 · For educational use only."
)

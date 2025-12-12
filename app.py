import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Cataract Detection",
    page_icon="👁",
    layout="centered"
)

# ================================
# CUSTOM CSS STYLE
# ================================
st.markdown(
"""
<style>
body {
    background-color: #f5f5f7;
}

.title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #2b2b2b;
    margin-bottom: -10px;
}

.subtext {
    text-align: center;
    font-size: 16px;
    color: #666;
}

.upload-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    margin-top: 20px;
}

.result-box {
    background: white;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #ddd;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    margin-top: 20px;
}
</style>
""",
    unsafe_allow_html=True
)

# ================================
# CONFIG
# ================================
MODEL_PATH = "models/cataract_vit_model.keras"
LABEL_PATH = "labels.json"
IMG_SIZE = (224, 224)

# ================================
# LOAD LABELS
# ================================
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        label_map = json.load(f)
else:
    label_map = {"0": "cataract", "1": "normal"}

label_map = {int(k): v for k, v in label_map.items()}

# ================================
# LOAD MODEL
# ================================
st.markdown("<h1 class='title'>👁 Cataract Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Upload foto mata untuk mendeteksi apakah terdapat katarak.</p>", unsafe_allow_html=True)

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ================================
# PREPROCESSING
# ================================
def preprocess_image(pil_img):
    img = pil_img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# ================================
# PREDICT FUNCTION
# ================================
def predict_image(pil_img):
    arr = preprocess_image(pil_img)
    preds = model.predict(arr)[0]
    return preds


# ================================
# STREAMLIT UI
# ================================
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload gambar mata (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================================
# RUN PREDICTION
# ================================
if uploaded_file and st.button("🔎 Prediksi"):
    with st.spinner("Menganalisis gambar..."):

        preds = predict_image(image)
        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id] * 100)
        label = label_map[class_id]

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📌 Hasil Prediksi")
        st.write(f"**Label:** `{label}`")
        st.write(f"**Confidence:** `{confidence:.2f}%`")
        st.progress(min(100, int(confidence)))
        st.markdown("</div>", unsafe_allow_html=True)

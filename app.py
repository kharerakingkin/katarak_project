import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image
import os


# =====================================================
# KONFIGURASI GLOBAL
# =====================================================
MODEL_PATH = "models/cataract_vit_model.h5"
LABEL_PATH = "labels.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Cataract Detector",
    page_icon="👁",
    layout="centered"
)

# Styling UI
st.markdown("""
    <style>
        .main {
            background-color: #F5F7FA;
        }
        .title {
            text-align: center;
            font-size: 32px !important;
            font-weight: 800 !important;
        }
        .sub {
            text-align: center;
            font-size: 18px !important;
            color: #444;
        }
        .result-box {
            padding: 20px;
            border-radius: 15px;
            background: white;
            border: 1px solid #ddd;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD LABELS
# =====================================================
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        LABELS = json.load(f)
else:
    LABELS = {0: "cataract", 1: "normal"}


# =====================================================
# LOAD MODEL
# =====================================================
st.markdown("<h1 class='title'>🔍 Cataract Detection System</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sub'>Upload gambar mata untuk dianalisis</p>",
            unsafe_allow_html=True)

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success(f"Model berhasil dimuat: `{MODEL_PATH}`")
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")


# =====================================================
# HAAR CASCADE
# =====================================================
FACE_CASCADE = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")


def contains_eye(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(img, 1.2, 5)
    eyes = EYE_CASCADE.detectMultiScale(img, 1.2, 4)
    return len(faces) > 0 or len(eyes) > 0


# =====================================================
# PREPROCESSING
# =====================================================
def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# =====================================================
# PREDIKSI
# =====================================================
def predict(img):
    arr = preprocess(img)
    preds = model.predict(arr)[0]
    return preds


# =====================================================
# GRAD-CAM
# =====================================================
def generate_gradcam(pil_img):

    img_array = preprocess(pil_img)

    # cari layer conv terakhir
    last_conv = None
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            last_conv = layer
            break

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_output)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    heatmap = cv2.resize(heatmap, (pil_img.width, pil_img.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    return cv2.addWeighted(np.array(pil_img), 0.6, heatmap, 0.4, 0)


# =====================================================
# STREAMLIT UI
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    if st.button("🔎 Analisis Gambar"):
        with st.spinner("Menganalisis gambar..."):

            # cek mata
            if not contains_eye(image):
                st.warning("⚠ Tidak ditemukan mata pada gambar.")
                st.stop()

            preds = predict(image)
            class_id = int(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

            # final label
            label = LABELS[str(class_id)]

            # tampilkan hasil
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("📌 Hasil Prediksi")
            st.write(f"**Label:** `{label}`")
            st.write(f"**Confidence:** `{confidence:.2f}%`")
            st.progress(int(confidence))
            st.markdown("</div>", unsafe_allow_html=True)

            # Grad-CAM
            if st.checkbox("🔥 Tampilkan Grad-CAM"):
                heat = generate_gradcam(image)
                if heat is not None:
                    st.image(heat, caption="Grad-CAM", use_column_width=True)
                else:
                    st.error("Grad-CAM tidak tersedia untuk model ini.")

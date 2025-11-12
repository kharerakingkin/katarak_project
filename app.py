import os
import io
import json
import glob
import numpy as np
from PIL import Image
from datetime import datetime
from collections import Counter

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Deteksi Katarak AI", page_icon="👁️", layout="wide")

# ==============================
# PATHS & CONSTANTS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CASCADE_DIR = os.path.join(BASE_DIR, "cascades")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85
EMBED_DIM = 576  # penting agar cocok dengan TransformerBlock

# ==============================
# CSS (fallback jika style.css tidak ada)
# ==============================
def apply_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body{font-family:Inter, sans-serif;}
        .main-header{font-size:2.2em;text-align:center;margin-bottom:0.2em;}
        .subheader{font-size:1.05em;text-align:center;color:#666;margin-bottom:1em;}
        .disclaimer{background:#fffae6;padding:12px;border-left:4px solid #ffc107;border-radius:8px;margin-bottom:1em;color:#664d03;}
        div.stButton > button{width:100%;padding:8px;border-radius:8px;}
        </style>
        """, unsafe_allow_html=True)
apply_css()

# ==============================
# Custom TransformerBlock
# ==============================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=max(1, self.embed_dim // max(1, num_heads)),
                                             output_shape=self.embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(self.embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ==============================
# Helper: cari file terbaru
# ==============================
def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# ==============================
# Load Model & Cascade
# ==============================
@st.cache_resource
def load_model_and_weights():
    # cari file model .keras
    model_file = latest_file(os.path.join(MODEL_DIR, "cataract_model_final_*.keras")) \
              or latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.keras")) \
              or os.path.join(MODEL_DIR, "cataract_model_final_20251112-142204.keras")

    if not os.path.exists(model_file):
        return None, f"Model file not found in {MODEL_DIR}. Expected .keras file."

    try:
        model = tf.keras.models.load_model(model_file, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
    except Exception as e:
        return None, f"Failed to load model: {e}"

    # cari bobot .weights.h5
    weight_file = latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.weights.h5")) \
               or latest_file(os.path.join(MODEL_DIR, "cataract_weights_*.weights.h5")) \
               or os.path.join(MODEL_DIR, "cataract_model_best.weights.h5")

    if os.path.exists(weight_file):
        try:
            model.load_weights(weight_file)
            weight_msg = f"Weights loaded from {os.path.basename(weight_file)}"
        except Exception as e:
            weight_msg = f"Found weights but failed to load: {e}"
    else:
        weight_msg = "No external weights found (using embedded model weights)."

    return model, f"Model loaded: {os.path.basename(model_file)} | {weight_msg}"

@st.cache_resource
def load_cascades():
    face_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
    if not (os.path.exists(face_path) and os.path.exists(eye_path)):
        return None, None, "Cascade XML not found — autozoom disabled."

    face_cascade = cv2.CascadeClassifier(face_path)
    eye_cascade = cv2.CascadeClassifier(eye_path)
    if face_cascade.empty() or eye_cascade.empty():
        return None, None, "Cascade files invalid."
    return face_cascade, eye_cascade, "Cascade loaded successfully."

# ==============================
# Load resources
# ==============================
model, model_msg = load_model_and_weights()
FACE_CASCADE, EYE_CASCADE, cascade_msg = load_cascades()

# ==============================
# Load Labels
# ==============================
labels_path = os.path.join(MODEL_DIR, "labels.json")
if os.path.exists(labels_path):
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    except Exception:
        labels = {"0": "normal", "1": "cataract"}
else:
    labels = {"0": "normal", "1": "cataract"}

# ==============================
# Image utilities
# ==============================
def preprocess_image_for_model(pil_image):
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_quality_heuristic(pil_image):
    gray = pil_image.convert("L").resize((128, 128))
    arr = np.array(gray).astype(np.float32) / 255.0
    gy, gx = np.gradient(arr)
    sharpness = np.sqrt(gx ** 2 + gy ** 2).var()
    mean_brightness = arr.mean()
    return not (sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95)

def crop_to_eye(pil_image, face_cascade, eye_cascade):
    if face_cascade is None or eye_cascade is None:
        return pil_image
    try:
        opencv_image = np.array(pil_image.convert('RGB'))
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            return pil_image
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        if len(eyes) == 0:
            return pil_image
        ex, ey, ew, eh = eyes[0]
        eye_x_center = x + ex + ew // 2
        eye_y_center = y + ey + eh // 2
        zoom_size = int(max(ew, eh) * 2.5)
        final_x_start = max(0, eye_x_center - zoom_size // 2)
        final_y_start = max(0, eye_y_center - zoom_size // 2)
        final_x_end = min(pil_image.width, eye_x_center + zoom_size // 2)
        final_y_end = min(pil_image.height, eye_y_center + zoom_size // 2)
        if final_x_end <= final_x_start or final_y_end <= final_y_start:
            return pil_image
        return pil_image.crop((final_x_start, final_y_start, final_x_end, final_y_end))
    except Exception:
        return pil_image

# ==============================
# UI HEADER
# ==============================
st.markdown("<h1 class='main-header'>👁️ Deteksi Katarak Berbasis AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Menggunakan OpenCV Auto Zoom untuk fokus ke area mata</p>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ Hasil ini hanya indikasi awal. Konsultasikan dengan dokter mata profesional.</div>", unsafe_allow_html=True)

if model is None:
    st.error(f"❌ Model tidak tersedia. {model_msg}")
else:
    st.success(f"✅ {model_msg}")

st.info(cascade_msg)

# ==============================
# MAIN UI
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata")
    uploaded_file = st.file_uploader("Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file and model:
        image_bytes = uploaded_file.read()
        pil_img_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_img_original, caption="Gambar Asli", use_column_width=True)

        if st.button("🚀 Mulai Prediksi"):
            with st.spinner("Menganalisis gambar..."):
                try:
                    pil_img_final = crop_to_eye(pil_img_original, FACE_CASCADE, EYE_CASCADE)
                    if pil_img_final != pil_img_original:
                        st.success("✅ Auto Zoom berhasil mendeteksi area mata.")
                        st.image(pil_img_final, caption="Gambar Mata (Crop)", use_column_width=True)

                    if not image_quality_heuristic(pil_img_final):
                        st.warning("⚠️ Gambar buram/gelap/terlalu terang. Coba unggah ulang.")
                        st.stop()

                    X = preprocess_image_for_model(pil_img_final)
                    preds = model.predict(X, verbose=0)[0]
                    top_idx = int(np.argmax(preds))
                    top_conf = float(preds[top_idx])
                    predicted_label = labels.get(str(top_idx), "unknown")

                    st.write("#### Probabilitas per kelas:")
                    for i, v in enumerate(preds):
                        lbl = labels.get(str(i), str(i))
                        st.write(f"- **{lbl.capitalize()}**: {v*100:.2f}%")

                    if top_conf < CONFIDENCE_THRESHOLD:
                        st.warning(f"Kepercayaan rendah ({top_conf*100:.2f}%)")
                    elif predicted_label == "normal":
                        st.success(f"✅ Mata terdeteksi NORMAL — {top_conf*100:.2f}%")
                    elif predicted_label == "cataract":
                        st.error(f"⚠️ Indikasi KATARAK — {top_conf*100:.2f}%")
                    else:
                        st.info(f"Hasil: {predicted_label} ({top_conf*100:.2f}%)")

                except Exception as e:
                    st.error(f"Gagal prediksi: {e}")

with col2:
    st.markdown("## ℹ️ Informasi Katarak")
    st.write("""
    Katarak adalah kekeruhan pada lensa mata yang menyebabkan penglihatan kabur.  
    **Gejala umum:**  
    - Penglihatan buram atau berkabut  
    - Warna tampak pudar  
    - Silau saat melihat cahaya terang  
    - Kesulitan melihat di malam hari  
    """)


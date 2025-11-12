import os
import io
import json
import glob
import time
from datetime import datetime

import numpy as np
from collections import Counter
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# OpenCV: use headless in requirements (opencv-python-headless)
import cv2

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Deteksi Katarak AI", page_icon="👁️", layout="wide")

# ==============================
# CONSTANTS & PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CASCADE_DIR = os.path.join(BASE_DIR, "cascades")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85
EMBED_DIM = 576  # penting agar custom TransformerBlock cocok saat load model

# ==============================
# EMBEDDED CSS (fallback jika style.css tidak ada)
# ==============================
def apply_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # fallback CSS embedded
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
# CUSTOM TRANSFORMER (needed to load model saved with it)
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
# HELPERS: cari file model/weights terbaru
# ==============================
def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

# ==============================
# LOAD MODEL & CASCADES (cached)
# ==============================
@st.cache_resource
def load_model_and_weights():
    # cari model .keras terbaru (support timestamped names)
    model_file = latest_file(os.path.join(MODEL_DIR, "cataract_model_final_*.keras")) \
                 or latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.keras")) \
                 or os.path.join(MODEL_DIR, "cataract_model_best.keras")

    if not model_file or not os.path.exists(model_file):
        return None, f"Model file not found in {MODEL_DIR}. Expected .keras file."

    try:
        model = tf.keras.models.load_model(model_file, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
    except Exception as e:
        return None, f"Failed to load model: {e}"

    # optional: cari bobot .weights.h5 terbaru dan muat jika ada
    weight_file = latest_file(os.path.join(MODEL_DIR, "cataract_weights_*.weights.h5")) \
                  or latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.weights.h5")) \
                  or os.path.join(MODEL_DIR, "cataract_weights.weights.h5")

    if weight_file and os.path.exists(weight_file):
        try:
            model.load_weights(weight_file)
            weight_msg = f"Weights loaded from {os.path.basename(weight_file)}"
        except Exception as e:
            weight_msg = f"Found weights but failed to load: {e}"
    else:
        weight_msg = "No extra weights file found (using model internal weights)."

    return model, weight_msg

@st.cache_resource
def load_cascades():
    face_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
    if not (os.path.exists(face_path) and os.path.exists(eye_path)):
        return None, None, "Cascade XML files not found (autozoom disabled)."

    try:
        face_c = cv2.CascadeClassifier(face_path)
        eye_c = cv2.CascadeClassifier(eye_path)
        if face_c.empty() or eye_c.empty():
            return None, None, "Cascade loaded but classifiers empty (invalid files)."
        return face_c, eye_c, "Cascades loaded."
    except Exception as e:
        return None, None, f"OpenCV cascade load error: {e}"

# ==============================
# init resources
# ==============================
model, model_msg = load_model_and_weights()
FACE_CASCADE, EYE_CASCADE, cascade_msg = load_cascades()

# ==============================
# LOAD LABELS
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
# SMALL UTIL FUNCTIONS
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
    if sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95:
        return False
    return True

def crop_to_eye(pil_image, face_cascade, eye_cascade):
    if face_cascade is None or eye_cascade is None:
        return pil_image

    try:
        opencv_image = np.array(pil_image.convert('RGB'))
        opencv_image_bgr = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return pil_image

        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
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

        cropped_eye_pil = pil_image.crop((final_x_start, final_y_start, final_x_end, final_y_end))
        return cropped_eye_pil
    except Exception:
        return pil_image

# ==============================
# UI HEADER & STATUS MESSAGES
# ==============================
st.markdown("<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Menggunakan OpenCV Auto Zoom untuk fokus ke mata</p>", unsafe_allow_html=True)
st.markdown(f"<div class='disclaimer'>⚠️ Disclaimer: Hasil hanya indikator. Konsultasikan ke profesional medis.</div>", unsafe_allow_html=True)

if model is None:
    st.error(f"❌ Model tidak tersedia. {model_msg}")
else:
    st.info(f"Model siap. {model_msg}")

if "Cascade" in cascade_msg or "loaded" in cascade_msg:
    st.success(cascade_msg)
else:
    st.warning(cascade_msg)

# ==============================
# MAIN LAYOUT
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata")
    uploaded_file = st.file_uploader("Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image_bytes = uploaded_file.read()
            pil_img_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(pil_img_original, caption="Gambar Asli", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal buka gambar: {e}")
            pil_img_original = None

        if st.button("🚀 Mulai Prediksi") and pil_img_original is not None:
            if model is None:
                st.error("Model belum tersedia. Tidak dapat melakukan prediksi.")
            else:
                with st.spinner("Analisis..."):
                    try:
                        # Auto zoom jika cascade ada
                        if FACE_CASCADE is not None and EYE_CASCADE is not None:
                            pil_img_cropped = crop_to_eye(pil_img_original, FACE_CASCADE, EYE_CASCADE)
                            if pil_img_cropped != pil_img_original:
                                st.success("✅ Area mata berhasil di-zoom.")
                                st.image(pil_img_cropped, caption="Crop Mata (Input Model)", use_column_width=True)
                                pil_img_final = pil_img_cropped
                            else:
                                st.info("ℹ️ Auto-zoom tidak melakukan crop — menggunakan gambar asli.")
                                pil_img_final = pil_img_original
                        else:
                            pil_img_final = pil_img_original
                            st.warning("⚠️ Auto-zoom tidak aktif. Menggunakan gambar asli.")

                        # quality check
                        if not image_quality_heuristic(pil_img_final):
                            st.warning("⚠️ Gambar dianggap kurang berkualitas (buram/terlalu terang/gelap). Coba unggah ulang.")
                            st.stop()

                        # preprocessing & predict
                        X = preprocess_image_for_model(pil_img_final)
                        preds = model.predict(X, verbose=0)[0]
                        preds = np.array(preds).astype(float)

                        top_idx = int(np.argmax(preds))
                        top_conf = float(preds[top_idx])

                        if top_conf < CONFIDENCE_THRESHOLD:
                            st.warning(f"Kepercayaan terlalu rendah ({top_conf*100:.2f}%). Unggah gambar yang lebih jelas.")
                            st.stop()

                        predicted_label = labels.get(str(top_idx), "unknown")
                        st.write("#### Probabilitas per kelas:")
                        for i, v in enumerate(preds):
                            lbl = labels.get(str(i), str(i))
                            st.write(f"- **{lbl.capitalize()}**: {v*100:.2f}%")

                        if predicted_label == "normal":
                            st.success(f"✅ Prediksi: NORMAL — {top_conf*100:.2f}%")
                        elif predicted_label == "cataract":
                            st.error(f"⚠️ Prediksi: INDIKASI KATARAK — {top_conf*100:.2f}%")
                        else:
                            st.info(f"Hasil: {predicted_label} — {top_conf*100:.2f}%")

                    except Exception as e:
                        st.error(f"Kesalahan saat prediksi: {e}")

with col2:
    st.markdown("## ℹ️ Informasi Katarak")
    st.write("""
    Katarak adalah kekeruhan pada lensa mata yang menyebabkan penglihatan kabur. 
    Gejala: penglihatan buram, silau, warna pudar, kesulitan di malam hari.
    Faktor risiko: usia lanjut, diabetes, merokok, paparan UV, riwayat cedera.
    """)

import os
import io
import json
import time
import numpy as np
from collections import Counter

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ==========================
# CONFIG
# ==========================
st.set_page_config(layout="wide", page_title="Deteksi Katarak AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "models", "cataract_model_best.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "cataract_weights.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85
EMBED_DIM = 576

# ==========================
# CUSTOM CSS STYLING
# ==========================
custom_css = """
body {
    font-family: 'Inter', sans-serif;
}
.main-header {
    text-align: center;
    margin-bottom: 0.5em;
    font-size: 2.5em;
    animation: slideInUp 0.8s ease-out;
}
.subheader {
    text-align: center;
    font-size: 1.2em;
    color: #888;
    margin-bottom: 2em;
    animation: slideInUp 1s ease-out;
}
.disclaimer {
    background-color: #fffae6;
    padding: 15px;
    border-left: 5px solid #ffc107;
    border-radius: 8px;
    margin-bottom: 2em;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    color: #664d03;
    animation: fadeIn 1.2s ease-in;
}
.disclaimer b {
    color: #ff9800;
}
.section-header {
    font-size: 1.8em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
    animation: fadeIn 1.5s ease-in;
}
div.stButton > button {
    width: 100%;
    padding: 10px;
    border-radius: 8px;
    font-size: 1.1em;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    animation: pulse 1.5s infinite;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.stAlert {
    animation: bounceIn 0.8s ease-out;
}
@media (max-width: 768px) {
    .main-header { font-size: 2em; }
    .subheader { font-size: 1em; }
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes bounceIn {
    0% { transform: scale(0.3); opacity: 0; }
    50% { transform: scale(1.1); opacity: 1; }
    80% { transform: scale(0.9); }
    100% { transform: scale(1); }
}
@keyframes slideInUp {
  0% { transform: translateY(20px); opacity: 0; }
  100% { transform: translateY(0); opacity: 1; }
}
"""
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# ==========================
# HEADER & DISCLAIMER
# ==========================
st.markdown("<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Sistem Deteksi Dini Katarak Menggunakan Kecerdasan Buatan</p>", unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> Hasil ini hanya sebagai referensi awal. 
Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat. 
Akurasi bergantung pada kualitas gambar mata yang diunggah.
</div>
""", unsafe_allow_html=True)

# ==========================
# TransformerBlock (harus sama dengan train.py)
# ==========================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.embed_dim // max(1, num_heads),
            output_shape=self.embed_dim
        )
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

# ==========================
# Model Loader
# ==========================
@st.cache_resource
def load_model_cached():
    if os.path.exists(MODEL_KERAS_PATH):
        try:
            model = tf.keras.models.load_model(
                MODEL_KERAS_PATH, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    st.error("Model tidak ditemukan di folder models/.")
    return None

model = load_model_cached()

# ==========================
# Load Labels
# ==========================
def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            return json.load(f)
    return {"0": "cataract", "1": "normal"}

labels = load_labels()

# ==========================
# Helper Functions
# ==========================
def preprocess_image_for_model(pil_image: Image.Image):
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def image_quality_heuristic(pil_image: Image.Image):
    gray = pil_image.convert("L").resize((128, 128))
    arr = np.array(gray).astype(np.float32) / 255.0
    gy, gx = np.gradient(arr)
    grad_mag = np.sqrt(gx**2 + gy**2)
    sharpness = grad_mag.var()
    mean_brightness = arr.mean()
    return not (sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95)

# ==========================
# UI Layout
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file and model:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_img, caption="Gambar Mata yang Diunggah", use_container_width=True)

        if st.button("🚀 Mulai Prediksi"):
            with st.spinner("Analisis sedang berlangsung..."):
                try:
                    if not image_quality_heuristic(pil_img):
                        st.warning("⚠️ Gambar terlalu buram, gelap, atau tidak fokus. Harap unggah gambar mata yang lebih jelas.")
                    else:
                        X = preprocess_image_for_model(pil_img)
                        preds = model.predict(X, verbose=0)[0]
                        top_idx = int(np.argmax(preds))
                        top_conf = float(preds[top_idx])
                        classes = {k: v for k, v in labels.items()}
                        prob_dict = {classes[str(i)]: float(preds[i]) * 100 for i in range(len(preds))}

                        if top_conf < CONFIDENCE_THRESHOLD:
                            st.warning(f"**Gambar Tidak Valid atau Tidak Relevan** — kepercayaan tertinggi {top_conf*100:.2f}%")
                            st.json({k: f"{v:.2f}%" for k, v in prob_dict.items()})
                        else:
                            predicted_label = classes[str(top_idx)]
                            if predicted_label == "normal":
                                st.success(f"Mata terdeteksi **NORMAL** ({top_conf*100:.2f}%)")
                            else:
                                st.error(f"Indikasi **KATARAK** ({top_conf*100:.2f}%)")

                            st.write("### 📊 Probabilitas per kelas:")
                            for cls, pct in prob_dict.items():
                                st.write(f"- **{cls}**: {pct:.2f}%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")

with col2:
    st.markdown("## 📚 Informasi Katarak", unsafe_allow_html=True)
    st.write("""
Katarak adalah kondisi di mana lensa mata menjadi keruh, menyebabkan penglihatan kabur atau buram.

**Gejala Umum:**
- Penglihatan kabur, warna pudar
- Sensitif terhadap cahaya
- Kesulitan melihat di malam hari

**Faktor Risiko:**
- Usia lanjut
- Diabetes
- Merokok
- Paparan sinar UV berlebih
""")

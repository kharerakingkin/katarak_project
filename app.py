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
import scipy.stats  # <== pastikan sudah diinstall: pip install scipy

# ==========================
# CONFIG
# ==========================
st.set_page_config(layout="wide", page_title="Deteksi Katarak AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "models", "cataract_model_best.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.95
EMBED_DIM = 576

# ==========================
# Styling
# ==========================
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with open("style.css", "w") as f:
    f.write("""
    body { font-family: 'Inter', sans-serif; }
    .main-header { text-align: center; margin-bottom: 0.5em; font-size: 2.5em; animation: slideInUp 0.8s ease-out; }
    .subheader { text-align: center; font-size: 1.2em; color: #888; margin-bottom: 2em; animation: slideInUp 1s ease-out; }
    .disclaimer { background-color: #fffae6; padding: 15px; border-left: 5px solid #ffc107; border-radius: 8px; margin-bottom: 2em; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #664d03; animation: fadeIn 1.2s ease-in; }
    div.stButton > button { width: 100%; padding: 10px; border-radius: 8px; font-size: 1.1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease; animation: pulse 1.5s infinite; }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    @keyframes slideInUp { 0% { transform: translateY(20px); opacity: 0; } 100% { transform: translateY(0); opacity: 1; } }
    """)
local_css("style.css")

# ==========================
# Transformer Block
# ==========================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embed_dim // max(1, num_heads), output_shape=self.embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(self.embed_dim)])
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
# Build Hybrid Model (backup)
# ==========================
def build_hybrid_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    x = base_model(inputs, training=False)
    x = layers.Reshape((-1, x.shape[-1]))(x)
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_embed = layers.Embedding(input_dim=x.shape[1], output_dim=x.shape[2])(positions)
    x = x + pos_embed
    x = TransformerBlock(num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(num_heads=4, ff_dim=128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)

# ==========================
# Load Model & Labels
# ==========================
@st.cache_resource
def load_model_cached():
    try:
        model = tf.keras.models.load_model(MODEL_KERAS_PATH, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model_cached()
try:
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
except Exception:
    labels = {"0": "cataract", "1": "normal"}

# ==========================
# Helper Functions
# ==========================
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
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    sharpness = grad_mag.var()
    mean_brightness = arr.mean()
    if sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95:
        return False
    return True

# ==========================
# Streamlit UI
# ==========================
st.markdown("<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Kini sistem mampu menolak gambar bukan mata 🧠</p>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ <b>Disclaimer:</b> Hasil ini hanya indikasi awal. Konsultasikan dengan dokter mata.</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata")
    uploaded_file = st.file_uploader("Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file and model:
        image_bytes = uploaded_file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_img, caption="Gambar yang Diunggah", use_container_width=True)

        if st.button("🚀 Mulai Prediksi"):
            with st.spinner("Analisis sedang berlangsung..."):
                try:
                    # 1️⃣ Cek kualitas dasar
                    if not image_quality_heuristic(pil_img):
                        st.warning("⚠️ Gambar buram/terlalu terang — bukan gambar mata valid.")
                        st.stop()

                    # 2️⃣ Prediksi model
                    X = preprocess_image_for_model(pil_img)
                    preds = model.predict(X, verbose=0)[0]
                    preds = np.array(preds).astype(float)

                    # 3️⃣ Entropy + Confidence filter
                    entropy = float(scipy.stats.entropy(preds))
                    top_idx = int(np.argmax(preds))
                    top_conf = float(preds[top_idx])

                    # 4️⃣ Tolak jika tidak relevan
                    if entropy > 0.8 or top_conf < CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ Gambar **tidak dikenali sebagai mata manusia**")
                        st.stop()

                    # 5️⃣ Hasil klasifikasi
                    predicted_label = labels.get(str(top_idx), "unknown")
                    prob_dict = {labels[str(i)]: float(preds[i]) * 100.0 for i in range(len(preds))}

                    if predicted_label == "normal":
                        st.success(f"Mata terdeteksi: **NORMAL** — {top_conf*100:.2f}%")
                    elif predicted_label == "cataract":
                        st.error(f"Mata terdeteksi: **INDIKASI KATARAK** — {top_conf*100:.2f}%")
                    else:
                        st.info(f"Hasil: {predicted_label} — {top_conf*100:.2f}%")

                    st.write("#### Probabilitas per kelas:")
                    for cls, pct in prob_dict.items():
                        st.write(f"- **{cls}**: {pct:.2f}%")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")

with col2:
    st.markdown("## 📚 Informasi Katarak")
    st.write("""
    Katarak adalah kondisi mata di mana lensa mata menjadi keruh, menyebabkan penglihatan kabur atau buram. Biasanya terjadi pada orang lanjut usia. Ini adalah penyebab utama kebutaan yang dapat diobati di seluruh dunia.
    
    **Gejala Umum:**
    - Penglihatan kabur, buram, atau berkabut
    - Warna terlihat pudar atau kurang jelas
    - Silau yang mengganggu saat melihat cahaya terang (terutama malam hari)
    - Peningkatan penglihatan dekat sementara pada lansia (disebut 'second sight')
    - Kesulitan melihat di malam hari atau dalam cahaya redup
    
    **Faktor Risiko:**
    - Usia (>60 tahun)
    - Diabetes
    - Merokok
    - Paparan sinar UV berlebihan tanpa perlindungan
    - Riwayat cedera mata atau peradangan sebelumnya
    - Penggunaan steroid jangka panjang
    """)

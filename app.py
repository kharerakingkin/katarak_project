import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import streamlit as st

# ==========================
# KONFIGURASI
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cataract_model_latest.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ==========================
# DEFINISI CUSTOM LAYER
# ==========================
@keras.saving.register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim // self.num_heads
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout)
        self.dropout2 = layers.Dropout(self.dropout)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ==========================
# LOAD MODEL & LABELS
# ==========================
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"TransformerBlock": TransformerBlock}
        )
    except Exception as e:
        st.error(f"⚠️ Gagal memuat model: {e}")
        st.stop()

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

model = load_model()
labels = load_labels()

# ==========================
# FUNGSI PREDIKSI
# ==========================
def predict(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# ==========================
# UI
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# CSS BARU — KARTU HASIL LEBIH JELAS DAN KONTRAS
st.markdown("""
<style>
.result-card {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    padding: 25px 35px;
    margin-top: 40px;
    color: #222;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}
.result-icon {
    font-size: 60px;
    margin-right: 25px;
}
.result-text h2 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
}
.result-text p {
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: 16px;
    color: #444;
}
.stats {
    margin-top: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.stat {
    background-color: #f3f3f3;
    padding: 8px 14px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 15px;
}
.cataract-card {
    border-left: 6px solid #d32f2f;
}
.normal-card {
    border-left: 6px solid #2e7d32;
}
.irrelevant-card {
    border-left: 6px solid #f9a825;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<h2 style='text-align:center; color:#00897b;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center; color:#555;'>Unggah foto mata Anda untuk mendeteksi indikasi katarak menggunakan model MobileNetV3 + Vision Transformer.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", width=350)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds) * 100
            predicted_class = classes[np.argmax(preds)]

            if confidence < 90:
                predicted_class = "irrelevant"

        # ==========================
        # TAMPILKAN HASIL
        # ==========================
        if predicted_class.lower() == "cataract":
            icon, title, desc, card_class = "⚠️", "Indikasi Katarak", "Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut.", "cataract-card"
        elif predicted_class.lower() == "normal":
            icon, title, desc, card_class = "✅", "Mata Normal", "Tidak ditemukan tanda-tanda katarak. Tetap jaga kesehatan mata Anda.", "normal-card"
        else:
            icon, title, desc, card_class = "❓", "Gambar Tidak Relevan", "Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus.", "irrelevant-card"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-text">
                <h2>{title}</h2>
                <p>{desc}</p>
                <div class="stats">
                    <div class="stat">Cataract: {cataract_prob:.2f}%</div>
                    <div class="stat">Normal: {normal_prob:.2f}%</div>
                    <div class="stat">Confidence: {confidence:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

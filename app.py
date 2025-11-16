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
# REGISTER CUSTOM LAYER
# ==========================
try:
    register_serializable = keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=embed_dim // self.num_heads
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
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model tidak ditemukan!")
        st.stop()

    try:
        return tf.keras.models.load_model(
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
# KONFIGURASI STREAMLIT
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# Sidebar Tema
st.sidebar.title("🎨 Pengaturan Tampilan")
theme = st.sidebar.radio("Pilih Tema:", ["🌞 Light Mode", "🌙 Dark Mode"])

# Warna tema
if "Dark" in theme:
    bg_color = "#1e1e1e"
    text_color = "#f1f1f1"
    card_bg = "#2a2a2a"
    shadow = "rgba(255,255,255,0.1)"
else:
    bg_color = "#f9f9f9"
    text_color = "#222"
    card_bg = "#ffffff"
    shadow = "rgba(0,0,0,0.1)"

# ==========================
# CSS
# ==========================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
.result-card {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    background-color: {card_bg};
    border-radius: 14px;
    box-shadow: 0 4px 12px {shadow};
    padding: 25px 35px;
    margin-top: 30px;
}}
.result-icon {{
    font-size: 55px;
    margin-right: 20px;
}}
.result-text h2 {{
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: {text_color};
}}
.result-text p {{
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: 15px;
    color: {text_color};
}}
.stats {{
    margin-top: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}}
.stat {{
    background-color: rgba(0,0,0,0.08);
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 600;
    color: {text_color};
}}
.cataract-card {{ border-left: 6px solid #ef5350; }}
.normal-card {{ border-left: 6px solid #66bb6a; }}
.irrelevant-card {{ border-left: 6px solid #fbc02d; }}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown(f"""
<h2 style='text-align:center; color:{text_color};'>
    👁️ Aplikasi Deteksi Katarak Berbasis AI
</h2>
<p style='text-align:center; color:{text_color}; opacity:0.8;'>
    Unggah foto mata Anda untuk mendeteksi indikasi katarak menggunakan model AI MobileNetV3 + Vision Transformer.
</p>
""", unsafe_allow_html=True)

# ==========================
# UPLOAD GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", width=320)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("🔎 Menganalisis gambar..."):

            # Hasil prediksi (2 kelas)
            preds = predict(image)
            classes = list(labels.values())

            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100

            predicted_class = classes[np.argmax(preds)]
            confidence = np.max(preds) * 100

            # Aturan deteksi gambar tidak relevan
            if confidence < 70:  # bisa dinaikkan jika ingin lebih ketat
                predicted_class = "irrelevant"

        # ==========================
        # KARTU HASIL
        # ==========================
        if predicted_class.lower() == "cataract":
            icon, title, desc, card_class = (
                "⚠️", 
                "Indikasi Katarak", 
                "Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut.",
                "cataract-card"
            )
        elif predicted_class.lower() == "normal":
            icon, title, desc, card_class = (
                "✅", 
                "Mata Normal",
                "Tidak ditemukan tanda-tanda katarak.",
                "normal-card"
            )
        else:
            icon, title, desc, card_class = (
                "❓",
                "Gambar Tidak Relevan",
                "Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas.",
                "irrelevant-card"
            )

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

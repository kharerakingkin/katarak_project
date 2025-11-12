import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# ==========================
# KONFIGURASI
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cataract_model_latest.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ==========================
# LOAD MODEL & LABELS
# ==========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model tidak ditemukan di folder 'models/'.")
        st.stop()
    try:
        # Tambahkan custom_objects agar bisa load model hybrid
        from tensorflow.keras import layers

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
                self.ffn = tf.keras.Sequential([
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

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={"TransformerBlock": TransformerBlock}
        )
        return model

    except Exception as e:
        st.error(f"⚠️ Gagal memuat model: {e}")
        st.stop()

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error("❌ File labels.json tidak ditemukan di folder 'models/'.")
        st.stop()
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

model = load_model()
labels = load_labels()

# ==========================
# PREDIKSI
# ==========================
def predict(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# ==========================
# UI
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="centered")

# ========== STYLING ========== #
st.markdown("""
<style>
/* Background */
body {
    background-color: #f5f7fa;
}

/* Result Card */
.result-card {
    background-color: #ffffff;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    padding: 25px 35px;
    margin-top: 35px;
    color: #1a1a1a;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.result-icon {
    font-size: 55px;
    text-align: center;
}
.result-text h2 {
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 5px;
}
.result-text p {
    text-align: center;
    color: #444;
    font-size: 15px;
    margin-bottom: 12px;
}
.stats {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}
.stat {
    background-color: #f0f0f0;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
}

/* Color themes */
.cataract-card { border-top: 6px solid #d32f2f; }
.normal-card { border-top: 6px solid #388e3c; }
.irrelevant-card { border-top: 6px solid #f9a825; }

h2.title {
    text-align: center;
    color: #00897b;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<h2 class='title'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center; color:#555;'>
Unggah foto mata Anda untuk mendeteksi indikasi katarak menggunakan model MobileNetV3 + Transformer.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", width=350)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("🔎 Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds) * 100
            predicted_class = classes[np.argmax(preds)]

            if confidence < 90:
                predicted_class = "irrelevant"

        # ==========================
        # CARD HASIL BARU
        # ==========================
        if predicted_class.lower() == "cataract":
            icon, title, desc, card_class = "⚠️", "Indikasi Katarak", \
                "Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut.", "cataract-card"
        elif predicted_class.lower() == "normal":
            icon, title, desc, card_class = "✅", "Mata Normal", \
                "Tidak ditemukan tanda-tanda katarak. Tetap jaga kesehatan mata Anda.", "normal-card"
        else:
            icon, title, desc, card_class = "❓", "Gambar Tidak Relevan", \
                "Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus.", "irrelevant-card"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-text">
                <h2>{title}</h2>
                <p>{desc}</p>
                <div class="stats">
                    <div class="stat" style="color:#d32f2f;">Cataract: {cataract_prob:.2f}%</div>
                    <div class="stat" style="color:#388e3c;">Normal: {normal_prob:.2f}%</div>
                    <div class="stat">Confidence: {confidence:.2f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

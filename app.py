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
        st.error("❌ Model tidak ditemukan!")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
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
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# CSS BARU — DESAIN FLAT CARD YANG JELAS
st.markdown("""
<style>
.result-card {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    background-color: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    padding: 25px 35px;
    margin-top: 40px;
    color: #222;
}
.result-icon {
    font-size: 60px;
    margin-right: 25px;
}
.result-text h2 {
    margin: 0;
    font-size: 26px;
    font-weight: 700;
}
.result-text p {
    margin-top: 6px;
    margin-bottom: 10px;
    font-size: 16px;
    color: #444;
}
.stats {
    margin-top: 10px;
    display: flex;
    gap: 15px;
}
.stat {
    background-color: #f1f1f1;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
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
        # CARD HASIL BARU
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

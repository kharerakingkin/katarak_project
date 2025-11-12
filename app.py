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
        st.error("❌ Model tidak ditemukan! Pastikan file cataract_model_latest.keras ada di folder 'models/'.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

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
# UI STREAMLIT
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# CSS BARU — Lebih kontras dan modern
st.markdown("""
<style>
body {
    font-family: 'Poppins', sans-serif;
}
.result-card {
    border-radius: 16px;
    padding: 25px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transition: transform 0.2s ease-in-out;
}
.result-card:hover {
    transform: scale(1.02);
}
.cataract {
    background-color: #ffe6e6;
    color: #b30000;
    border-left: 8px solid #ff4b4b;
}
.normal {
    background-color: #e6ffec;
    color: #007a33;
    border-left: 8px solid #00c853;
}
.irrelevant {
    background-color: #fff7e6;
    color: #a66a00;
    border-left: 8px solid #ffb300;
}
.metric-container {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 15px;
    flex-wrap: wrap;
}
.metric-box {
    background-color: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 600;
    min-width: 150px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style='text-align:center; color:#00BFA6;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center; color:gray;'>Unggah gambar mata untuk mendeteksi indikasi katarak menggunakan model MobileNetV3 + Vision Transformer.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", use_container_width=False, width=400)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds) * 100
            predicted_class = classes[np.argmax(preds)]

            # Jika confidence < 90%, dianggap gambar tidak relevan
            if confidence < 90:
                predicted_class = "irrelevant"

        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        # Tentukan warna & pesan berdasarkan hasil
        if predicted_class.lower() == "cataract":
            css_class = "cataract"
            title = "⚠️ Indikasi KATARAK"
            desc = "Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut."
        elif predicted_class.lower() == "normal":
            css_class = "normal"
            title = "✅ Mata NORMAL"
            desc = "Tidak terdeteksi tanda-tanda katarak. Tetap jaga kesehatan mata."
        else:
            css_class = "irrelevant"
            title = "❓ Gambar Tidak Relevan"
            desc = "⚠️ Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus."

        # ==========================
        # TAMPILAN HASIL BARU
        # ==========================
        st.markdown(f"""
        <div class='result-card {css_class}'>
            <h2 style='margin-bottom:10px;'>{title}</h2>
            <p style='font-size:17px; line-height:1.6; margin-bottom:20px;'>{desc}</p>
            <div class='metric-container'>
                <div class='metric-box' style='color:#b30000;'>Cataract: {cataract_prob:.2f}%</div>
                <div class='metric-box' style='color:#007a33;'>Normal: {normal_prob:.2f}%</div>
                <div class='metric-box' style='color:#555;'>Confidence: {confidence:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

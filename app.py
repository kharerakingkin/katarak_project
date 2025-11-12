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

# CSS BARU (bersih dan kontras tinggi)
st.markdown("""
<style>
body {
    font-family: 'Poppins', sans-serif;
}
.result-container {
    margin-top: 30px;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    text-align: center;
}
.cataract {
    background-color: #ffebee;
    border: 3px solid #e53935;
    color: #b71c1c;
}
.normal {
    background-color: #e8f5e9;
    border: 3px solid #43a047;
    color: #1b5e20;
}
.irrelevant {
    background-color: #fff8e1;
    border: 3px solid #fbc02d;
    color: #8d6e00;
}
.result-icon {
    font-size: 60px;
    margin-bottom: 10px;
}
.result-title {
    font-size: 26px;
    font-weight: 700;
}
.result-desc {
    font-size: 16px;
    margin-bottom: 20px;
}
.prob-box {
    display: inline-block;
    margin: 10px;
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
    color: white;
}
.prob-cataract {
    background-color: #e53935;
}
.prob-normal {
    background-color: #43a047;
}
.prob-confidence {
    background-color: #757575;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<h2 style='text-align:center; color:#00BFA6;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center; color:#666;'>Unggah foto mata untuk mendeteksi indikasi katarak menggunakan model MobileNetV3 + Vision Transformer.</p>
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

            # Jika confidence < 90%, dianggap gambar tidak relevan
            if confidence < 90:
                predicted_class = "irrelevant"

        st.markdown("---")

        # ==========================
        # DESAIN BARU HASIL ANALISIS
        # ==========================
        if predicted_class.lower() == "cataract":
            icon = "⚠️"
            title = "Indikasi KATARAK"
            desc = "Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut."
            box_class = "cataract"
        elif predicted_class.lower() == "normal":
            icon = "✅"
            title = "Mata NORMAL"
            desc = "Tidak ditemukan tanda-tanda katarak. Tetap jaga kesehatan mata Anda!"
            box_class = "normal"
        else:
            icon = "❓"
            title = "Gambar Tidak Relevan"
            desc = "Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus."
            box_class = "irrelevant"

        st.markdown(f"""
        <div class="result-container {box_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-title">{title}</div>
            <div class="result-desc">{desc}</div>
            <div>
                <span class="prob-box prob-cataract">Cataract: {cataract_prob:.2f}%</span>
                <span class="prob-box prob-normal">Normal: {normal_prob:.2f}%</span>
                <span class="prob-box prob-confidence">Confidence: {confidence:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

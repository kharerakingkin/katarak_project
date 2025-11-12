import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import json

# ==========================
# KONFIGURASI
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cataract_model_latest.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
CONFIDENCE_THRESHOLD = 0.90  # ambang batas gambar tidak relevan

# ==========================
# LOAD MODEL & LABELS
# ==========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model tidak ditemukan! Pastikan file 'cataract_model_latest.keras' ada di folder 'models/'.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_data
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error("❌ File 'labels.json' tidak ditemukan di folder 'models/'.")
        st.stop()
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

model = load_model()
labels = load_labels()

# ==========================
# SIDEBAR MODE TAMPILAN
# ==========================
st.sidebar.title("⚙️ Pengaturan")
theme_choice = st.sidebar.radio("🎨 Mode Tampilan", ["🌙 Dark Mode", "☀️ Light Mode"])

if theme_choice == "🌙 Dark Mode":
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    accent_color = "#00BFA6"
    box_bg = "linear-gradient(145deg, #1E1E1E, #171717)"
else:
    bg_color = "#FFFFFF"
    text_color = "#222222"
    accent_color = "#00BFA6"
    box_bg = "#F8F9FA"

# ==========================
# CSS DINAMIS
# ==========================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}
h2, h4, h5, p, label {{
    color: {text_color} !important;
    font-family: 'Inter', sans-serif;
}}
.stButton>button {{
    background-color: {accent_color};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: 0.2s;
}}
.stButton>button:hover {{
    background-color: #02d8bd;
}}
.stProgress > div > div > div {{
    background-color: {accent_color} !important;
}}
.result-box {{
    background: {box_bg};
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    text-align: center;
    margin-top: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ==========================
# FUNGSI PREDIKSI
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

st.markdown(f"""
<h2 style='text-align:center; color:{accent_color};'>👁️ Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center;'>Unggah gambar mata untuk mendeteksi indikasi katarak menggunakan model <b>MobileNetV3 + Transformer</b>.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang diunggah", use_column_width=True)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("🧠 Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds)
            predicted_class = classes[np.argmax(preds)]

        # ==========================
        # KEPUTUSAN
        # ==========================
        if confidence < CONFIDENCE_THRESHOLD:
            status_label = "❓ Gambar Tidak Relevan"
            status_desc = "⚠️ Gambar tidak dikenali sebagai mata manusia. Pastikan mengunggah foto mata yang jelas."
            status_color = "#FFA500"
        elif predicted_class.lower() == "cataract":
            status_label = "⚠️ Indikasi Katarak"
            status_desc = "Model mendeteksi kemungkinan katarak. Konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut."
            status_color = "#FF4B4B"
        else:
            status_label = "✅ Normal"
            status_desc = "Tidak terdeteksi tanda-tanda katarak. Mata tampak normal."
            status_color = "#4BB543"

        # ==========================
        # TAMPILKAN HASIL
        # ==========================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧠 Klasifikasi", value=status_label)
        with col2:
            st.metric(label="📈 Tingkat Keyakinan", value=f"{confidence*100:.2f}%")

        st.progress(float(confidence))

        st.markdown(f"""
        <div class='result-box'>
            <h4 style='color:{status_color};'>{status_label}</h4>
            <p style='font-size:16px;'>{status_desc}</p>
            <hr style='margin:10px 0; border: 1px solid #333;'>
            <p><b style='color:#FF4B4B;'>Cataract:</b> {cataract_prob:.2f}%<br>
            <b style='color:#4BB543;'>Normal:</b> {normal_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

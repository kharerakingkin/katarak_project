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

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model tidak ditemukan! Pastikan file cataract_model_latest.keras ada di folder 'models/'.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

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
    preds = model.predict(img_array)[0]
    return preds

# ==========================
# UI STREAMLIT
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

st.markdown("""
<h2 style='text-align:center; color:#00BFA6;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center;'>Unggah gambar mata untuk memeriksa indikasi katarak menggunakan model MobileNetV3 + Transformer.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            predicted_class = classes[np.argmax(preds)]

        # Tampilkan hasil dengan desain menarik
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧠 Klasifikasi", value=predicted_class.upper())
        with col2:
            st.metric(label="📈 Akurasi Prediksi", value=f"{np.max(preds) * 100:.2f}%")

        st.progress(float(np.max(preds)))

        st.markdown(f"""
        <div style='padding:15px; border-radius:10px; background-color:#f9f9f9; text-align:center;'>
        <h4>Probabilitas per kelas:</h4>
        <b style='color:#FF4B4B;'>Cataract:</b> {cataract_prob:.2f}%<br>
        <b style='color:#4BB543;'>Normal:</b> {normal_prob:.2f}%<br><br>
        <span style='font-size:18px; color:#333;'>
        {"⚠️ <b>Indikasi KATARAK</b> — segera konsultasi ke dokter!" if predicted_class.lower() == "cataract" else "✅ <b>Tidak terdeteksi katarak</b> — mata tampak normal."}
        </span>
        </div>
        """, unsafe_allow_html=True)

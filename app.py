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

st.markdown("""
<h2 style='text-align:center; color:#00BFA6;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center;'>Unggah gambar mata untuk memeriksa indikasi katarak menggunakan model MobileNetV3 + Transformer.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Gambar yang diunggah", use_container_width=True)

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

        # ==========================
        # TAMPILAN HASIL
        # ==========================
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧠 Klasifikasi", value=predicted_class.upper())
        with col2:
            st.metric(label="📈 Tingkat Keyakinan", value=f"{confidence:.2f}%")

        st.progress(float(confidence / 100))

        # Warna hasil yang berbeda tergantung kelas
        if predicted_class.lower() == "cataract":
            color_bg = "#ffe5e5"
            color_text = "#b30000"
            message = "⚠️ <b>Indikasi KATARAK</b><br>Segera konsultasikan ke dokter mata."
        elif predicted_class.lower() == "normal":
            color_bg = "#e8ffec"
            color_text = "#006600"
            message = "✅ <b>Tidak terdeteksi katarak</b><br>Mata tampak normal."
        else:
            color_bg = "#fff3cd"
            color_text = "#856404"
            message = "❓ <b>Gambar Tidak Relevan</b><br>⚠️ Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus."

        # Tampilan hasil akhir
        st.markdown(f"""
        <div style='padding:18px; border-radius:12px; 
                    background-color:{color_bg}; 
                    color:{color_text}; 
                    text-align:center; 
                    font-size:18px; 
                    line-height:1.6; 
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.1); 
                    margin-top:15px;'>
            {message}
            <hr style='border:1px solid rgba(0,0,0,0.05); margin:10px 0;'>
            <b style='color:#b30000;'>Cataract:</b> {cataract_prob:.2f}% &nbsp;&nbsp;
            <b style='color:#006600;'>Normal:</b> {normal_prob:.2f}%
        </div>
        """, unsafe_allow_html=True)

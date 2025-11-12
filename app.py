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

st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# ==========================
# LOAD MODEL DAN LABEL
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
# FUNGSI PREDIKSI
# ==========================
def predict(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds

# ==========================
# UI UTAMA
# ==========================
st.markdown("""
<h2 style='text-align:center; color:#00BFA6;'>👁️ Aplikasi Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center;'>Unggah gambar mata untuk memeriksa indikasi katarak menggunakan model <b>MobileNetV3 + Transformer</b>.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Tampilkan gambar dalam ukuran proporsional (tidak full screen)
    st.image(image, caption="Gambar yang diunggah", use_container_width=False, width=300)

    if st.button("🔍 Deteksi Katarak"):
        with st.spinner("🧠 Menganalisis gambar..."):
            preds = predict(image)
            classes = list(labels.values())
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            predicted_class = classes[np.argmax(preds)]

            # Deteksi gambar tidak relevan jika prediksi tidak kuat
            confidence = np.max(preds) * 100
            if confidence < 90:
                predicted_class = "irrelevant"

        # ==========================
        # TAMPILKAN HASIL
        # ==========================
        st.markdown("---")
        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧠 Klasifikasi", value=predicted_class.upper())
        with col2:
            st.metric(label="📈 Akurasi Prediksi", value=f"{confidence:.2f}%")

        st.progress(float(np.max(preds)))

        # Dapatkan tema Streamlit (dark/light)
        theme = st.get_option("theme.base") or "light"
        is_dark = theme == "dark"

        # Warna adaptif
        def adaptive_color(light, dark):
            return dark if is_dark else light

        # Pilih warna berdasarkan hasil
        if predicted_class.lower() == "cataract":
            bg_color = adaptive_color("#ffe6e6", "#4d0000")
            text_color = adaptive_color("#b30000", "#ffb3b3")
            message = "⚠️ <b>Indikasi KATARAK</b> — segera konsultasi ke dokter!"
        elif predicted_class.lower() == "normal":
            bg_color = adaptive_color("#e8ffe8", "#003300")
            text_color = adaptive_color("#007a00", "#99ff99")
            message = "✅ <b>Tidak terdeteksi katarak</b> — mata tampak normal."
        else:  # Gambar tidak relevan
            bg_color = adaptive_color("#fff4e6", "#332200")
            text_color = adaptive_color("#cc6600", "#ffcc80")
            message = "❓ <b>Gambar Tidak Relevan</b><br>⚠️ Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus."

        # Tampilkan hasil dengan desain adaptif
        st.markdown(f"""
        <div style='
            padding:20px; 
            border-radius:12px; 
            background-color:{bg_color}; 
            text-align:center;
            border: 1px solid rgba(255,255,255,0.15);
            margin-top:10px;
        '>
            <h4 style='color:{text_color}; margin-bottom:10px;'>Probabilitas per kelas</h4>
            <b style='color:{adaptive_color("#d63031", "#ff7675")};'>Cataract:</b> {cataract_prob:.2f}%<br>
            <b style='color:{adaptive_color("#0984e3", "#74b9ff")};'>Normal:</b> {normal_prob:.2f}%<br><br>
            <span style='font-size:17px; color:{text_color};'>{message}</span>
        </div>
        """, unsafe_allow_html=True)

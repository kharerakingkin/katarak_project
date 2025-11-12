import os
import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import json
import plotly.graph_objects as go

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
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

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
st.sidebar.title("⚙️ Pengaturan Tampilan")
theme_choice = st.sidebar.radio("🎨 Pilih Mode", ["🌙 Dark Mode", "☀️ Light Mode"])

# Tema warna dinamis
if theme_choice == "🌙 Dark Mode":
    bg_color = "#0E1117"
    text_color = "#EDEDED"
    accent_color = "#00BFA6"
    card_bg = "linear-gradient(145deg, #1E1E1E, #171717)"
else:
    bg_color = "#FFFFFF"
    text_color = "#222222"
    accent_color = "#00BFA6"
    card_bg = "#F8F9FA"

# ==========================
# CSS RESPONSIVE
# ==========================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4, h5, h6, p, label {{
    color: {text_color} !important;
}}
.stButton>button {{
    background-color: {accent_color};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    transition: 0.3s;
}}
.stButton>button:hover {{
    background-color: #02d8bd;
}}
.result-card {{
    background: {card_bg};
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    text-align: center;
    margin-top: 20px;
    transition: all 0.3s ease-in-out;
}}
@media (max-width: 768px) {{
    .stColumn {{
        flex-direction: column !important;
    }}
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
<h2 style='text-align:center; color:{accent_color}; font-weight:700;'>👁️ Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center; font-size:17px;'>Unggah gambar mata untuk mendeteksi indikasi katarak menggunakan model <b>MobileNetV3 + Transformer</b>.</p>
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
            status_desc = "⚠️ Gambar tidak dikenali sebagai mata. Mohon unggah foto mata yang jelas dan fokus."
            status_color = "#FFA500"
        elif predicted_class.lower() == "cataract":
            status_label = "⚠️ Indikasi Katarak"
            status_desc = "Model mendeteksi adanya indikasi katarak. Segera konsultasikan ke dokter mata untuk pemeriksaan lebih lanjut."
            status_color = "#FF4B4B"
        else:
            status_label = "✅ Normal"
            status_desc = "Tidak terdeteksi tanda-tanda katarak. Mata tampak normal."
            status_color = "#4BB543"

        # ==========================
        # HASIL RESPONSIF
        # ==========================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🧠 Klasifikasi", value=status_label)
        with col2:
            st.metric(label="📈 Keyakinan", value=f"{confidence*100:.2f}%")

        # Bar Chart (Plotly)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["Cataract", "Normal"],
                    y=[cataract_prob, normal_prob],
                    marker_color=[status_color if p == np.max(preds)*100 else "#00BFA6" for p in [cataract_prob, normal_prob]],
                    text=[f"{cataract_prob:.1f}%", f"{normal_prob:.1f}%"],
                    textposition="outside"
                )
            ]
        )
        fig.update_layout(
            title="Probabilitas Klasifikasi",
            xaxis_title="Kelas",
            yaxis_title="Probabilitas (%)",
            template="plotly_dark" if theme_choice == "🌙 Dark Mode" else "plotly_white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

        # ==========================
        # KARTU HASIL
        # ==========================
        st.markdown(f"""
        <div class='result-card'>
            <h3 style='color:{status_color}; font-weight:700;'>{status_label}</h3>
            <p style='font-size:16px;'>{status_desc}</p>
            <hr style='margin:10px 0; border: 1px solid rgba(0,0,0,0.1);'>
            <p><b style='color:#FF4B4B;'>Cataract:</b> {cataract_prob:.2f}%<br>
            <b style='color:#4BB543;'>Normal:</b> {normal_prob:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

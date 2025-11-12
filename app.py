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
# DEFINISI CUSTOM LAYER
# ==========================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=embed_dim // self.num_heads,
            output_shape=embed_dim
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)
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
        st.error("❌ Model tidak ditemukan! Pastikan file 'cataract_model_latest.keras' ada di folder 'models/'.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)

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
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(page_title="Deteksi Katarak AI", layout="wide")

# ==========================
# DETEKSI AUTO THEME DARI SISTEM
# ==========================
st.markdown("""
<script>
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
window.parent.postMessage({themeMode: prefersDark ? 'dark' : 'light'}, '*');
</script>
""", unsafe_allow_html=True)

if "themeMode" not in st.session_state:
    st.session_state.themeMode = "light"

# Dapatkan sinyal dari browser
theme_input = st.query_params.get("theme", None)
if theme_input:
    st.session_state.themeMode = theme_input

# ==========================
# SIDEBAR UNTUK OVERRIDE
# ==========================
st.sidebar.title("⚙️ Pengaturan Tampilan")
theme_choice = st.sidebar.radio("🎨 Pilih Mode", ["🌙 Dark Mode", "☀️ Light Mode", "🌓 Otomatis (Sistem)"], index=2)

if theme_choice == "🌙 Dark Mode":
    theme = "dark"
elif theme_choice == "☀️ Light Mode":
    theme = "light"
else:
    theme = st.session_state.themeMode  # otomatis dari sistem

# ==========================
# WARNA BERDASARKAN THEME
# ==========================
if theme == "dark":
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
# CSS Styling
# ==========================
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
    font-family: 'Inter', sans-serif;
}}
h2, h3, h4, p, label {{
    color: {text_color} !important;
}}
.stButton>button {{
    background-color: {accent_color};
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}}
.stButton>button:hover {{
    background-color: #02d8bd;
    transform: scale(1.05);
}}
.result-card {{
    background: {card_bg};
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    text-align: center;
    margin-top: 20px;
}}
.uploaded-img {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 15px;
}}
.uploaded-img img {{
    max-width: 350px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: transform 0.4s ease-in-out;
}}
.uploaded-img img:hover {{
    transform: scale(1.12);
}}
@media (max-width: 600px) {{
    .uploaded-img img {{
        max-width: 250px;
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
st.markdown(f"""
<h2 style='text-align:center; color:{accent_color}; font-weight:700;'>👁️ Deteksi Katarak Berbasis AI</h2>
<p style='text-align:center;'>Unggah gambar mata untuk memeriksa indikasi katarak menggunakan model <b>MobileNetV3 + Transformer</b>.</p>
""", unsafe_allow_html=True)

if "reset" not in st.session_state:
    st.session_state.reset = False

if not st.session_state.reset:
    uploaded_file = st.file_uploader("📤 Unggah Gambar Mata", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
        st.image(image, caption="🖼️ Gambar yang diunggah", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Deteksi Katarak"):
            with st.spinner("🧠 Menganalisis gambar..."):
                preds = predict(image)
                classes = list(labels.values())
                cataract_prob = preds[0] * 100
                normal_prob = preds[1] * 100
                confidence = np.max(preds)
                predicted_class = classes[np.argmax(preds)]

            if confidence < CONFIDENCE_THRESHOLD:
                status_label = "❓ Gambar Tidak Relevan"
                status_desc = "⚠️ Gambar tidak dikenali sebagai mata. Harap unggah foto mata yang jelas dan fokus."
                status_color = "#FFA500"
            elif predicted_class.lower() == "cataract":
                status_label = "⚠️ Indikasi Katarak"
                status_desc = "Model mendeteksi indikasi katarak. Sebaiknya konsultasi ke dokter mata."
                status_color = "#FF4B4B"
            else:
                status_label = "✅ Normal"
                status_desc = "Tidak terdeteksi tanda-tanda katarak."
                status_color = "#4BB543"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Hasil Analisis")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🧠 Klasifikasi", value=status_label)
            with col2:
                st.metric(label="📈 Keyakinan", value=f"{confidence*100:.2f}%")

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
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class='result-card'>
                <h3 style='color:{status_color}; font-weight:700;'>{status_label}</h3>
                <p>{status_desc}</p>
                <hr>
                <b style='color:#FF4B4B;'>Cataract:</b> {cataract_prob:.2f}%<br>
                <b style='color:#4BB543;'>Normal:</b> {normal_prob:.2f}%
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### ")
            if st.button("🔁 Ulangi / Reset"):
                st.session_state.reset = True
                st.rerun()

else:
    st.session_state.reset = False
    st.rerun()

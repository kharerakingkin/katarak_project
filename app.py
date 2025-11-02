import os
import streamlit as st
import streamlit.components.v1 as components
import time
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json
import io

# Mengatur layout halaman ke mode "wide"
st.set_page_config(layout="wide")

# ==============================================================================
#                 KONFIGURASI PATH MODEL DAN PARAMETER
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_ABSOLUTE = os.path.join(BASE_DIR, "models", "cataract_model_best.keras")
LABELS_PATH_ABSOLUTE = os.path.join(BASE_DIR, "models", "labels.json")

CONFIDENCE_THRESHOLD = 0.85
EMBED_DIM = 576 
# ==============================================================================


# --- Custom CSS for Styling ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<style> .main-header {text-align: center; font-size: 2.5em;} </style>",
            unsafe_allow_html=True
        )

with open("style.css", "w") as f:
    f.write(
        """
    body { font-family: 'Inter', sans-serif; }
    .main-header { text-align: center; margin-bottom: 0.5em; font-size: 2.5em; animation: slideInUp 0.8s ease-out; }
    .subheader { text-align: center; font-size: 1.2em; color: #888; margin-bottom: 2em; animation: slideInUp 1s ease-out; }
    .disclaimer { background-color: #fffae6; padding: 15px; border-left: 5px solid #ffc107; border-radius: 8px; margin-bottom: 2em; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: #664d03; animation: fadeIn 1.2s ease-in; }
    .disclaimer b { color: #ff9800; }
    .section-header { font-size: 1.8em; margin-top: 1.5em; margin-bottom: 0.8em; animation: fadeIn 1.5s ease-in; }
    div.stButton > button { width: 100%; padding: 10px; border-radius: 8px; font-size: 1.1em; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease; animation: pulse 1.5s infinite; }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stAlert { animation: bounceIn 0.8s ease-out; }
    @media (max-width: 768px) { .main-header { font-size: 2em; } .subheader { font-size: 1em; } }
    @keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.02); } 100% { transform: scale(1); } }
    @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
    @keyframes bounceIn { 0% { transform: scale(0.3); opacity: 0; } 50% { transform: scale(1.1); opacity: 1; } 80% { transform: scale(0.9); } 100% { transform: scale(1); } }
    @keyframes slideInUp { 0% { transform: translateY(20px); opacity: 0; } 100% { transform: translateY(0); opacity: 1; } }
    """
    )
local_css("style.css")

# ==============================================================================
#                 TRANSFORMER BLOCK DENGAN PERBAIKAN value_shape
# ==============================================================================

@tf.keras.utils.register_keras_serializable() 
class TransformerBlock(keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim 
        self.rate = rate
        self.embed_dim = EMBED_DIM

        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embed_dim) 
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="relu"), 
             keras.layers.Dense(self.embed_dim)] 
        )
        
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    # METODE BUILD KRITIS DENGAN MEMPERBAIKI ARGUMEN value_shape
    def build(self, input_shape):
        embed_shape = (input_shape[0], input_shape[1], self.embed_dim)

        # PERBAIKAN KRITIS: Memanggil build dengan 3 shape (query, key, value)
        self.att.build(embed_shape, embed_shape, embed_shape) 

        self.ffn.build(embed_shape) 
        self.layernorm1.build(embed_shape)
        self.layernorm2.build(embed_shape)
        
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ==============================================================================
#                      FUNGSI LOAD MODEL MENGGUNAKAN PATH ABSOLUT
# ==============================================================================

@st.cache_resource
def load_model():
    model_path = MODEL_PATH_ABSOLUTE 
    
    # --- DEBUGGING PATH ---
    if not os.path.exists(model_path):
        st.error(f"❌ DEBUGGING GAGAL: File model tidak ditemukan di path: {model_path}")
        try:
            files_in_models = os.listdir(os.path.join(BASE_DIR, "models"))
            st.info(f"Isi folder models/ yang terlihat: {files_in_models}")
        except:
            st.info("Folder 'models' tidak dapat diakses.")
        return None
    # --- Akhir Debugging ---

    try:
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects={"TransformerBlock": TransformerBlock}
        )
        st.success("✅ Model AI berhasil dimuat!")
        return model
    except Exception as e:
        st.exception(f"❌ Error saat memuat model: {e}")
        st.error(
            "SOLUSI: Error Keras ini menunjukkan masalah versi atau koding kustom. "
            "Pastikan Anda telah memperbarui `requirements.txt` ke versi yang kompatibel."
        )
        return None


model = load_model()

# ... (Kode pemuatan label)
try:
    with open(LABELS_PATH_ABSOLUTE, "r") as f:
        labels = json.load(f)
except FileNotFoundError:
    st.error("Error: Label file (labels.json) tidak ditemukan.")
    labels = {"0": "normal", "1": "cataract"}
except Exception as e:
    st.error(f"Error saat memuat label: {e}")
    labels = {"0": "normal", "1": "cataract"}


# ==============================================================================
#                 ANTARMUKA STREAMLIT
# ==============================================================================

# --- Header Section ---
st.markdown(
    "<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subheader'>Sistem Deteksi Dini Katarak Menggunakan Kecerdasan Buatan</p>",
    unsafe_allow_html=True,
)

# --- Disclaimer Section ---
st.markdown(
    """
<div class="disclaimer">
⚠️ <b>Disclaimer:</b> Hasil ini hanya sebagai referensi awal. Selalu konsultasikan dengan dokter mata untuk diagnosis yang akurat. **Akurasi bergantung pada kualitas gambar mata yang diunggah.**
</div>
""",
    unsafe_allow_html=True,
)

# --- Usage Guide Section ---
st.markdown("## 📚 Panduan Penggunaan", unsafe_allow_html=True)
st.write(
    """Aplikasi ini dirancang untuk mendeteksi indikasi katarak dari gambar mata. Ikuti langkah-langkah mudah di bawah ini:

1. **Unggah Gambar:** Klik area `Pilih gambar mata` di bawah untuk mengunggah foto mata yang jelas.
2. **Kualitas Gambar:** Pastikan gambar **fokus pada mata** dan tidak buram.
3. **Mulai Analisis:** Klik tombol 🚀 `Mulai Prediksi` setelah gambar ditampilkan.
4. **Lihat Hasil:** Hasil akan menunjukkan **Normal**, **Indikasi Katarak**, atau **Gambar Tidak Valid/Tidak Relevan** jika gambar tidak jelas atau bukan gambar mata.
"""
)

# --- Main Columns for Upload and Information ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file and model:
        image_bytes = uploaded_file.read()
        st.image(
            image_bytes, caption="Gambar Mata yang Diunggah", use_container_width=True
        )

        prediction_result = None
        confidence_percent = 0.0

        if st.button("🚀 Mulai Prediksi"):
            with st.spinner("Analisis sedang berlangsung..."):
                try:
                    # Preprocess image
                    img = (
                        Image.open(io.BytesIO(image_bytes))
                        .convert("RGB")
                        .resize((224, 224))
                    )
                    img_array = np.array(img)

                    # Preprocessing sesuai model MobileNetV3
                    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(
                        img_array
                    )
                    img_array = np.expand_dims(img_array, axis=0)

                    # Predict
                    predictions = model.predict(img_array)

                    confidence_score = np.max(predictions[0])
                    confidence_percent = confidence_score * 100

                    predicted_class = np.argmax(predictions[0])
                    prediction = labels.get(str(predicted_class), "unknown")

                    time.sleep(1)

                    # --- Logika 3 Klasifikasi ---
                    if confidence_score < CONFIDENCE_THRESHOLD:
                        st.warning(
                            f"Hasil Prediksi: **Gambar Tidak Valid atau Tidak Relevan** (Kepercayaan tertinggi **{confidence_percent:.2f}%**). Harap unggah foto mata yang jelas dan relevan. Analisis ditolak karena keraguan model."
                        )
                    elif prediction == "normal":
                        st.success(
                            f"Hasil Prediksi: Mata terdeteksi **Normal** dengan tingkat kepercayaan **{confidence_percent:.2f}%**."
                        )
                    elif prediction == "cataract":
                        st.error(
                            f"Hasil Prediksi: Mata terdeteksi memiliki **indikasi Katarak** dengan tingkat kepercayaan **{confidence_percent:.2f}%**."
                        )
                    else:
                        st.exception(
                            f"Error: Label prediksi ({prediction}) tidak dikenali. Harap cek file labels.json."
                        )

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
    elif uploaded_file and not model:
        st.warning("Model AI belum berhasil dimuat. Tidak dapat melakukan prediksi.")


with col2:
    st.markdown("## 📚 Informasi Katarak", unsafe_allow_html=True)
    st.write(
        """
    Katarak adalah kondisi mata di mana lensa mata menjadi keruh, menyebabkan penglihatan kabur atau buram. Biasanya terjadi pada orang lanjut usia. Ini adalah penyebab utama kebutaan yang dapat diobati di seluruh dunia.
    
    **Gejala Umum:**
    - Penglihatan kabur, buram, atau berkabut
    - Warna terlihat pudar atau kurang jelas
    - Silau yang mengganggu saat melihat cahaya terang (terutama malam hari)
    - Peningkatan penglihatan dekat sementara pada lansia (disebut 'second sight')
    - Kesulitan melihat di malam hari atau dalam cahaya redup
    
    **Faktor Risiko:**
    - Usia (>60 tahun)
    - Diabetes
    - Merokok
    - Paparan sinar UV berlebihan tanpa perlindungan
    - Riwayat cedera mata atau peradangan sebelumnya
    - Penggunaan steroid jangka panjang
    """
    )
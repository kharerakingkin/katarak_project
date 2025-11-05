import os
import io
import json
import numpy as np

# Library Utama
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import scipy.stats 
import cv2 # Library OpenCV untuk Deteksi dan Auto Zoom

# ==========================
# KONFIGURASI DAN PATH
# ==========================
st.set_page_config(layout="wide", page_title="Deteksi Katarak AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "models", "cataract_model_best.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

# Path Haar Cascade
CASCADE_DIR = os.path.join(BASE_DIR, "cascades") 
FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.95
EMBED_DIM = 576

# ==========================
# STYLING (disederhanakan)
# ==========================
def local_css(file_name):
    """Memuat file CSS lokal."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Membuat dan memuat style.css
with open("style.css", "w") as f:
    f.write("""
    .main-header { text-align: center; font-size: 2.5em; }
    .subheader { text-align: center; font-size: 1.2em; color: #888; }
    .disclaimer { background-color: #fffae6; padding: 15px; border-left: 5px solid #ffc107; border-radius: 8px; }
    div.stButton > button { width: 100%; padding: 10px; border-radius: 8px; font-size: 1.1em; }
    """)
local_css("style.css")

# ==========================
# MODEL UTAMA (Tetap)
# ==========================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Blok Transformer untuk model Hybrid Vision."""
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM
        # ... (Inisialisasi lapisan lainnya)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=self.embed_dim // max(1, num_heads), output_shape=self.embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(self.embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # ... (Logika call tetap)
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ==========================
# LOAD RESOURCES
# ==========================

@st.cache_resource
def load_model_cached():
    """Memuat model Keras utama."""
    try:
        model = tf.keras.models.load_model(MODEL_KERAS_PATH, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_cascades_cached():
    """Memuat Haar Cascade Classifiers dan menangani kegagalan file."""
    try:
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        
        if face_cascade.empty() or eye_cascade.empty():
             # Ini akan memicu error jika file XML tidak ditemukan/rusak
             st.error("❌ Gagal memuat Haar Cascade XML. Pastikan file ada di folder 'cascades'.")
             return None, None
        return face_cascade, eye_cascade
    except Exception as e:
        st.error(f"❌ Error in OpenCV: {e}")
        return None, None

model = load_model_cached()
FACE_CASCADE, EYE_CASCADE = load_cascades_cached()

try:
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
except Exception:
    labels = {"0": "cataract", "1": "normal"}


# ==========================
# PREPROCESSING & CV FUNCTIONS
# ==========================

def preprocess_image_for_model(pil_image):
    """Menyiapkan gambar PIL untuk input model Keras."""
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_quality_heuristic(pil_image):
    """Cek dasar ketajaman dan kecerahan."""
    gray = pil_image.convert("L").resize((128, 128))
    arr = np.array(gray).astype(np.float32) / 255.0
    # Hitung variance dari gradien untuk mengukur ketajaman
    gy, gx = np.gradient(arr)
    sharpness = np.sqrt(gx ** 2 + gy ** 2).var()
    mean_brightness = arr.mean()
    
    # Batas heuristik: ketajaman rendah ATAU terlalu gelap ATAU terlalu terang
    if sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95:
        return False
    return True

def crop_to_eye(pil_image, face_cascade, eye_cascade):
    """Menggunakan Haar Cascade untuk mendeteksi mata dan memotong gambar."""
    if face_cascade is None or eye_cascade is None:
        return pil_image # Kembali jika cascade tidak dimuat

    try:
        # Konversi ke format array NumPy BGR (OpenCV)
        opencv_image = np.array(pil_image.convert('RGB')) 
        opencv_image_bgr = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR) 
        gray = cv2.cvtColor(opencv_image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            return pil_image # Gagal deteksi wajah

        # Ambil wajah terbesar (asumsi paling dominan)
        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0] 
        roi_gray = gray[y:y+h, x:x+w]
        
        # Deteksi Mata di dalam Wajah
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        if len(eyes) == 0:
            return pil_image # Gagal deteksi mata
            
        # Gunakan mata pertama yang terdeteksi
        (ex, ey, ew, eh) = eyes[0]
        
        # Hitung pusat mata relatif terhadap gambar asli
        eye_x_center = x + ex + ew // 2
        eye_y_center = y + ey + eh // 2
        
        # Tentukan ukuran kotak zoom (misalnya 2.5x ukuran mata terdeteksi)
        zoom_size = int(max(ew, eh) * 2.5) 
        
        # Hitung batas akhir dengan memastikan tidak keluar dari batas gambar asli
        final_x_start = max(0, eye_x_center - zoom_size // 2)
        final_y_start = max(0, eye_y_center - zoom_size // 2)
        final_x_end = min(pil_image.width, eye_x_center + zoom_size // 2)
        final_y_end = min(pil_image.height, eye_y_center + zoom_size // 2)
        
        # Pastikan batas valid
        if final_x_end <= final_x_start or final_y_end <= final_y_start:
             return pil_image # Koordinat invalid (misalnya jika zoom_size terlalu kecil)

        # Crop dan kembalikan gambar PIL
        cropped_eye_pil = pil_image.crop((final_x_start, final_y_start, final_x_end, final_y_end))
        return cropped_eye_pil

    except Exception as e:
        # Tangkap error OpenCV yang tidak terduga dan kembalikan gambar asli
        st.warning(f"⚠️ Error saat crop dengan OpenCV: {e}. Menganalisis gambar asli.")
        return pil_image 


# ==========================
# STREAMLIT UI & LOGIC
# ==========================

st.markdown("<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Menggunakan **OpenCV Auto Zoom** untuk fokus ke mata.</p>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ <b>Disclaimer:</b> Hasil ini hanya indikasi awal. Konsultasikan dengan dokter mata.</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata")
    uploaded_file = st.file_uploader("Pilih gambar mata (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    # Pengecekan awal Cascade:
    if not FACE_CASCADE or not EYE_CASCADE:
        st.error("‼️ Fungsionalitas Auto Zoom (OpenCV) **dinonaktifkan** karena kegagalan memuat file Cascade. Harap periksa folder `cascades`.")
    
    if uploaded_file and model:
        image_bytes = uploaded_file.read()
        pil_img_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        st.image(pil_img_original, caption="Gambar yang Diunggah (Asli)", width='stretch')

        if st.button("🚀 Mulai Prediksi"):
            with st.spinner("Analisis sedang berlangsung..."):
                try:
                    # 1. AUTO ZOOM (hanya jika Cascade dimuat)
                    if FACE_CASCADE and EYE_CASCADE:
                        pil_img_cropped = crop_to_eye(pil_img_original, FACE_CASCADE, EYE_CASCADE)
                        
                        if pil_img_cropped == pil_img_original:
                            st.warning("⚠️ Deteksi mata gagal. Menganalisis seluruh gambar asli (mungkin kurang akurat).")
                            pil_img_final = pil_img_original
                        else:
                            st.info("✅ Area mata berhasil di-zoom.")
                            st.image(pil_img_cropped, caption="Gambar Mata yang Di-zoom (Input Model)", width='stretch')
                            pil_img_final = pil_img_cropped
                    else:
                        pil_img_final = pil_img_original
                        st.warning("⚠️ Menggunakan gambar asli karena Auto Zoom tidak aktif.")

                    st.write("---")
                    
                    # 2. Cek kualitas dasar
                    if not image_quality_heuristic(pil_img_final):
                        st.warning("⚠️ Gambar buram/terlalu terang/gelap — harap unggah gambar yang lebih baik.")
                        st.stop()

                    # 3. Prediksi model
                    X = preprocess_image_for_model(pil_img_final)
                    preds = model.predict(X, verbose=0)[0]
                    preds = np.array(preds).astype(float)

                    # 4. Filter Entropy + Confidence
                    # entropy = float(scipy.stats.entropy(preds)) # Tidak digunakan di sini, tapi bagus untuk deteksi OOD
                    top_idx = int(np.argmax(preds))
                    top_conf = float(preds[top_idx])

                    if top_conf < CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ Kepercayaan prediksi terlalu rendah ({top_conf*100:.2f}%). Harap unggah gambar mata yang lebih jelas.")
                        st.stop()

                    # 5. Hasil klasifikasi
                    predicted_label = labels.get(str(top_idx), "unknown")
                    prob_dict = {labels[str(i)]: float(preds[i]) * 100.0 for i in range(len(preds))}

                    if predicted_label == "normal":
                        st.success(f"Mata terdeteksi: **NORMAL** — {top_conf*100:.2f}%")
                    elif predicted_label == "cataract":
                        st.error(f"Mata terdeteksi: **INDIKASI KATARAK** — {top_conf*100:.2f}%")
                    else:
                        st.info(f"Hasil: {predicted_label} — {top_conf*100:.2f}%")

                    st.write("#### Probabilitas per kelas:")
                    for cls, pct in prob_dict.items():
                        st.write(f"- **{cls.capitalize()}**: {pct:.2f}%")

                except Exception as e:
                    st.error(f"Terjadi kesalahan fatal saat prediksi: {e}")

with col2:
    st.markdown("## 📚 Informasi Katarak")
    st.write("""
    Katarak adalah kondisi mata di mana lensa mata menjadi keruh, menyebabkan penglihatan kabur atau buram.
    
    **Gejala Umum:**
    - Penglihatan kabur atau berkabut.
    - Warna terlihat pudar.
    - Silau saat melihat cahaya terang.
    - Kesulitan melihat di malam hari.
    """)
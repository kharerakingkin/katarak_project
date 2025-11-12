import os
import io
import json
import glob
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Deteksi Katarak AI", page_icon="👁️", layout="wide")

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CASCADE_DIR = os.path.join(BASE_DIR, "cascades")
STYLE_PATH = os.path.join(BASE_DIR, "style.css")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85
EMBED_DIM = 576

# ==============================
# CSS
# ==============================
def apply_css():
    if os.path.exists(STYLE_PATH):
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body{font-family:'Inter',sans-serif;}
        .main-header{font-size:2.2em;text-align:center;margin-bottom:0.2em;}
        .subheader{font-size:1.05em;text-align:center;color:#666;margin-bottom:1em;}
        .disclaimer{background:#fffae6;padding:12px;border-left:4px solid #ffc107;
        border-radius:8px;margin-bottom:1em;color:#664d03;}
        div.stButton > button{width:100%;padding:8px;border-radius:8px;}
        </style>
        """, unsafe_allow_html=True)
apply_css()

# ==============================
# CUSTOM LAYER
# ==============================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM
        self.att = layers.MultiHeadAttention(num_heads=num_heads,
                                             key_dim=max(1, self.embed_dim // max(1, num_heads)),
                                             output_shape=self.embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(self.embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ==============================
# HELPERS
# ==============================
def latest_file(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None

@st.cache_resource
def load_model_and_weights():
    model_file = latest_file(os.path.join(MODEL_DIR, "cataract_model_final_*.keras")) \
                 or latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.keras"))
    if not model_file or not os.path.exists(model_file):
        return None, f"Model file not found in {MODEL_DIR}. Expected .keras file."

    try:
        model = tf.keras.models.load_model(model_file, custom_objects={"TransformerBlock": TransformerBlock}, compile=False)
    except Exception as e:
        return None, f"Failed to load model: {e}"

    weight_file = latest_file(os.path.join(MODEL_DIR, "cataract_weights_*.weights.h5")) \
                  or latest_file(os.path.join(MODEL_DIR, "cataract_model_best*.weights.h5"))
    if weight_file and os.path.exists(weight_file):
        try:
            model.load_weights(weight_file)
            msg = f"Weights loaded from {os.path.basename(weight_file)}"
        except Exception as e:
            msg = f"Found weights but failed to load: {e}"
    else:
        msg = "No extra weights file found (using internal weights)."
    return model, msg

@st.cache_resource
def load_cascades():
    face_path = os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
    eye_path = os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
    if not (os.path.exists(face_path) and os.path.exists(eye_path)):
        return None, None, "Cascade XML files not found (autozoom disabled)."
    try:
        face_c = cv2.CascadeClassifier(face_path)
        eye_c = cv2.CascadeClassifier(eye_path)
        if face_c.empty() or eye_c.empty():
            return None, None, "Cascade invalid."
        return face_c, eye_c, "Cascades loaded."
    except Exception as e:
        return None, None, f"OpenCV cascade load error: {e}"

# ==============================
# LOAD RESOURCES
# ==============================
model, model_msg = load_model_and_weights()
FACE_CASCADE, EYE_CASCADE, cascade_msg = load_cascades()

labels_path = os.path.join(MODEL_DIR, "labels.json")
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
else:
    labels = {"0": "normal", "1": "cataract"}

# ==============================
# FUNCTIONS
# ==============================
def preprocess_image_for_model(pil_image):
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def image_quality_heuristic(pil_image):
    gray = pil_image.convert("L").resize((128, 128))
    arr = np.array(gray).astype(np.float32) / 255.0
    gy, gx = np.gradient(arr)
    sharpness = np.sqrt(gx ** 2 + gy ** 2).var()
    mean_brightness = arr.mean()
    return not (sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95)

def crop_to_eye(pil_image, face_cascade, eye_cascade):
    if face_cascade is None or eye_cascade is None:
        return pil_image
    try:
        opencv_image = np.array(pil_image.convert('RGB'))
        opencv_image_bgr = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        if len(faces) == 0:
            return pil_image
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
        if len(eyes) == 0:
            return pil_image
        ex, ey, ew, eh = eyes[0]
        eye_x, eye_y = x + ex + ew//2, y + ey + eh//2
        zoom = int(max(ew, eh) * 2.5)
        x1, y1 = max(0, eye_x - zoom//2), max(0, eye_y - zoom//2)
        x2, y2 = min(pil_image.width, eye_x + zoom//2), min(pil_image.height, eye_y + zoom//2)
        return pil_image.crop((x1, y1, x2, y2))
    except Exception:
        return pil_image

# ==============================
# UI HEADER
# ==============================
st.markdown("<h1 class='main-header'>👁️ <b>Deteksi Katarak Berbasis AI</b></h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Dengan Auto-Zoom Area Mata & Analisis Kecerdasan Buatan</p>", unsafe_allow_html=True)
st.markdown("<div class='disclaimer'>⚠️ Hasil hanya indikator. Konsultasikan dengan profesional medis untuk diagnosis.</div>", unsafe_allow_html=True)

if model is None:
    st.error(f"❌ Model tidak tersedia. {model_msg}")
else:
    st.info(f"✅ Model siap digunakan — {model_msg}")

if "loaded" in cascade_msg:
    st.success(cascade_msg)
else:
    st.warning(cascade_msg)

# ==============================
# MAIN LAYOUT
# ==============================
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🖼️ Unggah Gambar Mata")
    uploaded_file = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        st.image(pil_img, caption="Gambar Asli", use_column_width=True)

        if st.button("🚀 Mulai Prediksi"):
            if model is None:
                st.error("Model belum tersedia.")
            else:
                with st.spinner("🔍 Menganalisis gambar..."):
                    img_cropped = crop_to_eye(pil_img, FACE_CASCADE, EYE_CASCADE)
                    if img_cropped != pil_img:
                        st.success("✅ Area mata terdeteksi dan di-zoom otomatis.")
                        st.image(img_cropped, caption="Hasil Auto-Zoom", use_column_width=True)
                    else:
                        st.info("ℹ️ Menggunakan gambar asli (auto-zoom tidak aktif).")

                    if not image_quality_heuristic(img_cropped):
                        st.warning("⚠️ Gambar buram/gelap/terlalu terang, hasil bisa kurang akurat.")
                        st.stop()

                    X = preprocess_image_for_model(img_cropped)
                    preds = model.predict(X, verbose=0)[0]
                    top_idx = int(np.argmax(preds))
                    top_conf = float(preds[top_idx])
                    predicted_label = labels.get(str(top_idx), "unknown")

                    st.markdown("---")
                    st.markdown("### 🎯 Hasil Analisis Model")
                    for i, v in enumerate(preds):
                        lbl = labels.get(str(i), str(i)).capitalize()
                        st.markdown(f"**{lbl}**: {v*100:.2f}%")
                        st.progress(float(v))

                    st.markdown("<br>", unsafe_allow_html=True)
                    if predicted_label == "normal":
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,#a8e063,#56ab2f);
                        padding:25px;border-radius:15px;color:white;text-align:center;
                        font-size:1.3em;font-weight:600;box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
                            ✅ <b>Hasil Deteksi: NORMAL</b><br>
                            <small>Tingkat keyakinan: {top_conf*100:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)
                    elif predicted_label == "cataract":
                        st.markdown(f"""
                        <div style='background:linear-gradient(135deg,#ff6a00,#ee0979);
                        padding:25px;border-radius:15px;color:white;text-align:center;
                        font-size:1.3em;font-weight:600;box-shadow:0 4px 15px rgba(0,0,0,0.2);
                        animation:pulse 2s infinite;'>
                            ⚠️ <b>Indikasi Katarak Terdeteksi</b><br>
                            <small>Tingkat keyakinan: {top_conf*100:.2f}%</small>
                        </div>
                        <style>
                        @keyframes pulse {{
                            0% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.4); }}
                            70% {{ box-shadow: 0 0 0 20px rgba(255, 0, 0, 0); }}
                            100% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }}
                        }}
                        </style>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background:#3498db;padding:25px;border-radius:15px;
                        color:white;text-align:center;font-size:1.3em;font-weight:600;
                        box-shadow:0 4px 15px rgba(0,0,0,0.2);'>
                            ℹ️ <b>Hasil: {predicted_label.capitalize()}</b><br>
                            <small>Tingkat keyakinan: {top_conf*100:.2f}%</small>
                        </div>
                        """, unsafe_allow_html=True)

with col2:
    st.markdown("## 👁️ Tentang Katarak")
    st.write("""
    Katarak adalah kekeruhan pada lensa mata yang menyebabkan penglihatan buram.
    Gejala umum meliputi:
    - Penglihatan kabur atau ganda  
    - Warna tampak pudar  
    - Silau berlebihan  
    - Sulit melihat di malam hari  

    **Faktor risiko:** usia lanjut, diabetes, paparan UV, merokok, atau trauma mata.
    """)

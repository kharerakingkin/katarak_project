import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import io

# Mengatur konfigurasi halaman di awal
st.set_page_config(
    page_title="Cataract Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="auto"
)

# ------------------------------------------------------------
# Prefer tflite_runtime (lightweight) → fallback to tensorflow
# ------------------------------------------------------------
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except Exception:
        Interpreter = None

# ------------------------------------------------------------
# AUTO SELECT TFLITE MODEL
# ------------------------------------------------------------
MODEL_DIR = "tflite_models"
CANDIDATES = [
    "cataract_model_float16.tflite",
    "cataract_model_float32.tflite",
]

SELECTED_MODEL = None
for fname in CANDIDATES:
    fpath = os.path.join(MODEL_DIR, fname)
    if os.path.exists(fpath):
        SELECTED_MODEL = fpath
        break

if SELECTED_MODEL is None:
    st.error("❌ Tidak ada model TFLite yang ditemukan di folder `tflite_models/`.")
    st.stop()
st.sidebar.success(f"Model: **{os.path.basename(SELECTED_MODEL)}** terdeteksi.")

# ------------------------------------------------------------
# Load labels
# ------------------------------------------------------------
LABEL_PATH = "tflite_models/labels.json"
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        raw_labels = json.load(f)
    labels = {int(k): v for k, v in raw_labels.items()}
else:
    labels = {0: "cataract", 1: "normal"}
    st.sidebar.warning("labels.json tidak ditemukan. Menggunakan default 2 kelas.")


# ------------------------------------------------------------
# Load TFLite Interpreter
# ------------------------------------------------------------
if Interpreter is None:
    st.error("Tidak dapat memuat interpreter TFLite.")
    st.stop()


@st.cache_resource
def load_interpreter(path):
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


interpreter = load_interpreter(SELECTED_MODEL)
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


# ------------------------------------------------------------
# Preprocessing & Prediction
# ------------------------------------------------------------
IMG_SIZE = (224, 224)


def preprocess(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(pil_img):
    arr = preprocess(pil_img)

    if input_details["dtype"] in [np.uint8, np.int8]:
        scale, zero = input_details["quantization"]
        arr = (arr / scale + zero).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details["index"])[0]

    if output_details["dtype"] in [np.uint8, np.int8]:
        scale, zero = output_details["quantization"]
        output_data = scale * (output_data.astype("float32") - zero)

    return output_data.astype("float32")


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

# HEADER & DESKRIPSI
st.markdown("""
# 👁️ Cataract AI Detector
Analisis cepat gambar mata menggunakan Model Hybrid **MobileNetV3 + ViT-Tail** yang telah dikuantisasi (TFLite).
""")

st.markdown("---")

# TATA LETAK KOLOM UTAMA
col_upload, col_result = st.columns([1, 1])

# --- KOLOM UPLOAD ---
with col_upload:
    st.markdown("### 🖼️ Unggah Gambar Mata")
    uploaded = st.file_uploader(
        "Pilih file gambar (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar diunggah", width=350)
        
        if st.button("🚀 MULAI PREDIKSI & ANALISIS", use_container_width=True, type="primary"):
             st.session_state.run_prediction = True

    else:
        st.info("Silakan unggah gambar mata (misal: close-up iris) untuk memulai analisis.")
        st.session_state.run_prediction = False

# --- KOLOM HASIL ---
with col_result:
    st.markdown("### 🩺 Hasil Diagnosis AI")
    
    if uploaded and st.session_state.get('run_prediction', False):
        with st.spinner("Menganalisa fitur visual mata..."):
            try:
                probs = predict_image(img)
                pred_index = int(np.argmax(probs))
                conf = float(np.max(probs)) * 100
                pred_label = labels.get(pred_index, str(pred_index))
                
                # LOGIKA PENENTUAN HASIL DAN INTERPRETASI
                if 'cataract' in pred_label.lower():
                    result_color = "#E33D3D"  # Merah Kuat
                    emoji = "🚨"
                    header_text = "TIDAK NORMAL (POTENSI KATARAK)"
                    interpretation = "Model mendeteksi adanya kekeruhan lensa yang konsisten dengan Katarak. **Sangat dianjurkan** untuk segera konsultasi dan pemeriksaan lebih lanjut oleh Dokter Spesialis Mata (Ophthalmologist)."
                else:
                    result_color = "#35A352"  # Hijau Kuat
                    emoji = "✅"
                    header_text = "NORMAL (SAAT INI)"
                    interpretation = "Model mengklasifikasikan gambar mata sebagai Normal. Tidak ada indikasi Katarak yang terdeteksi secara otomatis oleh AI. Meskipun demikian, pemeriksaan rutin oleh profesional tetap penting."
                
                # TAMPILAN KARTU HASIL UTAMA
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 25px; border-radius: 12px; border: 2px solid {result_color};'>
                    <h2 style='color: {result_color}; margin-top: 0px;'>{emoji} {header_text}</h2>
                    <p style='font-size: 16px;'>Keyakinan Model: <b>{conf:.2f}%</b></p>
                    <p style='font-size: 14px; margin-top: 15px;'>{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # BAGIAN DISKLAIMER MEDIS TAMBAHAN
                st.write("---")
                st.markdown(f"**❗ Peringatan Medis:** Hasil dengan keyakinan di bawah 80% harus diperlakukan dengan sangat hati-hati.")

                # VISUALISASI PROBABILITAS
                st.markdown("#### 📊 Distribusi Probabilitas")
                data = {'Label': [labels.get(i, str(i)).upper() for i in range(len(probs))], 'Probabilitas (%)': probs * 100}
                st.bar_chart(data, x='Label', y='Probabilitas (%)')

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
        
        st.session_state.run_prediction = False
    
    elif not uploaded:
        st.warning("Menunggu unggahan gambar...")

st.markdown("---")
st.markdown("""
<p style='font-size: 12px; text-align: center; color: #777;'>
Model ini berfungsi sebagai alat *screening* awal dan **BUKAN** pengganti diagnosis, pemeriksaan, atau konsultasi medis oleh Dokter Spesialis Mata (Ophthalmologist). Keputusan pengobatan harus selalu didasarkan pada penilaian profesional.
</p>
<p style='font-size: 12px; text-align: center; color: #555; margin-top: 10px;'>
<hr style='border-top: 1px solid #ddd; margin: 5px 0;'>
**Pengembang:** Kharera Prabandaru (4611421136)
</p>
""", unsafe_allow_html=True)
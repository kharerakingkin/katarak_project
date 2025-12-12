import streamlit as st
import numpy as np
from PIL import Image
import json
import os
import io

# ------------------------------------------------------------
# Prefer tflite_runtime (lightweight) → fallback to tensorflow
# ------------------------------------------------------------
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter
    st.info("Using: tflite_runtime interpreter")
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        st.info("Using: tensorflow.lite Interpreter")
    except Exception:
        Interpreter = None

# ------------------------------------------------------------
# AUTO SELECT TFLITE MODEL
# ------------------------------------------------------------
MODEL_DIR = "tflite_models"
CANDIDATES = [
    "cataract_model_float16.tflite",   # PRIORITY 1
    "cataract_model_float32.tflite",   # FALLBACK
]

SELECTED_MODEL = None

for fname in CANDIDATES:
    fpath = os.path.join(MODEL_DIR, fname)
    if os.path.exists(fpath):
        SELECTED_MODEL = fpath
        break

if SELECTED_MODEL is None:
    st.error("❌ Tidak ada model TFLite yang ditemukan di folder `tflite_models/`.\n\n"
             "Harap masukkan file:\n"
             "- cataract_model_float16.tflite **atau**\n"
             "- cataract_model_float32.tflite")
    st.stop()

st.success(f"Model otomatis terdeteksi → **{SELECTED_MODEL}**")


# ------------------------------------------------------------
# Load labels
# ------------------------------------------------------------
LABEL_PATH = "models/labels.json"
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH, "r") as f:
        raw_labels = json.load(f)
    labels = {int(k): v for k, v in raw_labels.items()}
else:
    labels = {0: "cataract", 1: "normal"}
    st.warning("labels.json tidak ditemukan. Menggunakan default 2 kelas.")


# ------------------------------------------------------------
# Load TFLite Interpreter
# ------------------------------------------------------------
if Interpreter is None:
    st.error("Tidak dapat memuat interpreter TFLite (tflite-runtime / tensorflow).")
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
# Preprocessing
# ------------------------------------------------------------
IMG_SIZE = (224, 224)


def preprocess(img: Image.Image):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0  # MUST match training
    arr = np.expand_dims(arr, axis=0)
    return arr


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
def predict_image(pil_img):
    arr = preprocess(pil_img)

    # input quantization
    if input_details["dtype"] in [np.uint8, np.int8]:
        scale, zero = input_details["quantization"]
        arr = (arr / scale + zero).astype(input_details["dtype"])

    interpreter.set_tensor(input_details["index"], arr)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details["index"])[0]

    # dequantize
    if output_details["dtype"] in [np.uint8, np.int8]:
        scale, zero = output_details["quantization"]
        output_data = scale * (output_data.astype("float32") - zero)

    return output_data.astype("float32")


# ------------------------------------------------------------
# UI (Responsive)
# ------------------------------------------------------------
st.title("👁 Cataract Detector — TFLite (Auto Model Select)")
st.write("Unggah gambar mata untuk diprediksi menggunakan model MobileNetV3 + ViT-Tail (TFLite).")

uploaded = st.file_uploader("Upload gambar mata", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    if st.button("🔍 Prediksi"):
        with st.spinner("Menganalisa..."):
            probs = predict_image(img)
            pred = int(np.argmax(probs))
            conf = float(np.max(probs)) * 100
            label = labels.get(pred, str(pred))

        st.subheader(f"Hasil: **{label.upper()}**")
        st.write(f"Confidence: {conf:.2f}%")
        st.progress(int(conf))

        st.write("📊 Probabilitas detail:")
        for i, p in enumerate(probs):
            st.write(f"- {labels.get(i, i)}: {p*100:.2f}%")

else:
    st.info("Silakan unggah gambar terlebih dahulu.")

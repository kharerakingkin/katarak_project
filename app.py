import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import json
import os

# =====================================================
# LOAD LABELS
# =====================================================
LABELS = {"0": "cataract", "1": "normal"}

# =====================================================
# LOAD MODEL
# =====================================================
MODEL_PATH = "models/final_model.keras"   # FIX
model = None

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# =====================================================
# HAAR CASCADE (FACE + EYE)
# =====================================================
FACE_CASCADE = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier("haarcascade/haarcascade_eye.xml")

def contains_eye(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.2, 5)
    eyes = EYE_CASCADE.detectMultiScale(gray, 1.2, 4)

    return len(faces) > 0 or len(eyes) > 0


# =====================================================
# PREPROCESS GAMBAR UNTUK MODEL
# =====================================================
IMG_SIZE = (224, 224)

def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# =====================================================
# PREDIKSI LABEL
# =====================================================
def predict(img):
    if model is None:
        raise RuntimeError("Model belum dimuat.")
    arr = preprocess(img)
    preds = model.predict(arr)[0]
    return preds


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🔍 Eye Cataract Classifier + Eye Detector + Grad-CAM")
st.write("Unggah gambar mata untuk dideteksi.")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_column_width=True)

    if st.button("🔍 Deteksi"):
        with st.spinner("Menganalisis..."):

            # 1. CEK APAKAH GAMBAR MENGANDUNG MATA
            if not contains_eye(image):
                st.warning("⚠ Gambar tidak relevan (tidak terdeteksi mata/wajah)")
                st.stop()

            # 2. PREDIKSI MODEL
            preds = predict(image)
            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds) * 100

            # 3. TENTUKAN LABEL
            if confidence < 60:
                label = "irrelevant"
            else:
                label = LABELS[str(np.argmax(preds))]

            st.subheader("Hasil Prediksi:")
            st.write(f"**Prediksi**: {label}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))

            # 4. Grad-CAM
            if st.checkbox("Tampilkan Grad-CAM"):
                import tensorflow.keras.backend as K

                last_conv_layer = model.get_layer(index=-5)
                grad_model = tf.keras.models.Model(
                    [model.inputs],
                    [last_conv_layer.output, model.output]
                )

                img_arr = preprocess(image)
                with tf.GradientTape() as tape:
                    conv_output, preds_val = grad_model(img_arr)
                    class_idx = tf.argmax(preds_val[0])
                    loss = preds_val[:, class_idx]

                grads = tape.gradient(loss, conv_output)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

                heatmap = cv2.resize(
                    heatmap.numpy(), (image.width, image.height))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed = cv2.addWeighted(
                    np.array(image), 0.6, heatmap, 0.4, 0)
                st.image(superimposed, caption="Grad-CAM",
                         use_column_width=True)

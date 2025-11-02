import os
import io
import json
import time
import numpy as np
from collections import Counter

import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ==========================
# CONFIG
# ==========================
st.set_page_config(layout="wide", page_title="Deteksi Katarak AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "models", "cataract_model_best.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "cataract_weights.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.85  # jika confidence < ini -> tolak sebagai "gambar tidak relevan"
EMBED_DIM = 576  # harus sesuai output channel dari MobileNetV3Small (dipakai di TransformerBlock)

# ==========================
# Styling (optional)
# ==========================
st.markdown("<style>body { font-family: Inter, sans-serif; }</style>", unsafe_allow_html=True)

# ==========================
# TransformerBlock (registered) - harus sama dengan train.py
# ==========================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.embed_dim = EMBED_DIM

        # MultiHeadAttention: gunakan output_shape untuk kompatibilitas
        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.embed_dim // max(1, num_heads),
            output_shape=self.embed_dim
        )
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(self.embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def build(self, input_shape):
        # Build sublayers explicitly for safety
        embed_shape = (input_shape[0], input_shape[1], self.embed_dim)
        # Build attention with shapes (query, key, value)
        try:
            # Some TF versions accept three args, some accept one; using build to be explicit
            self.att.build(embed_shape)
        except Exception:
            # best-effort: still call build on ffn and norms
            pass
        self.ffn.build(embed_shape)
        self.layernorm1.build(embed_shape)
        self.layernorm2.build(embed_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return cfg

# ==========================
# Positional embedding helper
# ==========================
def add_position_embedding(x):
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_embed = layers.Embedding(input_dim=x.shape[1], output_dim=x.shape[2])(positions)
    return x + pos_embed

# ==========================
# Build same hybrid model (for fallback load_weights)
# ==========================
def build_hybrid_model(input_shape=(224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNetV3Small(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    x = base_model(inputs, training=False)
    x = layers.Reshape((-1, x.shape[-1]))(x)
    x = add_position_embedding(x)
    x = TransformerBlock(num_heads=4, ff_dim=128)(x)
    x = TransformerBlock(num_heads=4, ff_dim=128)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)

# ==========================
# Load labels
# ==========================
def load_labels():
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r") as f:
                labels = json.load(f)
            # ensure keys are str
            labels = {str(k): v for k, v in labels.items()}
            return labels
        except Exception:
            st.warning("labels.json rusak, akan gunakan default mapping.")
    # fallback default
    return {"0": "cataract", "1": "normal"}

labels = load_labels()

# ==========================
# Model loading with fallback: .keras -> weights.h5 -> None
# ==========================
@st.cache_resource
def load_model_cached():
    # Try load .keras model first
    if os.path.exists(MODEL_KERAS_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_KERAS_PATH,
                                               custom_objects={"TransformerBlock": TransformerBlock},
                                               compile=False)  # prevent loading optimizer
            return model, "keras"
        except Exception as e:
            st.warning(f"Warning: gagal load .keras model: {e}")

    # fallback to weights.h5 using architecture builder
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = build_hybrid_model()
            model.load_weights(WEIGHTS_PATH)
            return model, "weights"
        except Exception as e:
            st.error(f"Gagal load weights fallback: {e}")
            return None, None

    st.error("Model tidak ditemukan (cari models/cataract_model_best.keras atau models/cataract_weights.h5).")
    return None, None

model, model_source = load_model_cached()

# ==========================
# Helper: preprocess image
# ==========================
def preprocess_image_for_model(pil_image: Image.Image):
    img = pil_image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v3.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Optional heuristic to reject non-eye images (not perfect, helps basic filters)
def image_quality_heuristic(pil_image: Image.Image):
    # returns True if image likely OK (sharp enough and not too dark/bright)
    gray = pil_image.convert("L").resize((128, 128))
    arr = np.array(gray).astype(np.float32) / 255.0
    # sharpness (variance of laplacian)
    try:
        import cv2
        lap = cv2.Laplacian(np.array(gray), cv2.CV_64F)
        sharpness = lap.var()
    except Exception:
        # fallback simple gradient variance if cv2 missing
        gy, gx = np.gradient(arr)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        sharpness = grad_mag.var()
    mean_brightness = arr.mean()
    # checks (these are heuristic thresholds; adjust if too strict)
    if sharpness < 0.0005 or mean_brightness < 0.08 or mean_brightness > 0.95:
        return False
    return True

# ==========================
# Streamlit UI
# ==========================
st.title("🩺 Deteksi Katarak (MobileNetV3Small + ViT Tail)")
st.write("Upload foto mata (focus pada mata). Sistem memberi hasil dalam persen dan menolak gambar tidak relevan jika confidence rendah.")

col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Pilih gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # display image preview
        try:
            image_bytes = uploaded_file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(pil_img, caption="Gambar yang diupload", use_column_width=True)
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")
            pil_img = None

        if pil_img is not None:
            # optional quick heuristic check
            ok_quality = image_quality_heuristic(pil_img)
            if not ok_quality:
                st.warning("Gambar tampak buram/terlalu gelap/terlalu terang. Sistem mungkin menolak prediksi.")
            if model is None:
                st.error("Model belum dimuat, tidak bisa memprediksi.")
            else:
                if st.button("🚀 Mulai Prediksi"):
                    with st.spinner("Sedang memproses..."):
                        try:
                            # preprocess
                            X = preprocess_image_for_model(pil_img)
                            preds = model.predict(X, verbose=0)[0]
                            # ensure numerical stability
                            preds = np.array(preds).astype(float)
                            top_idx = int(np.argmax(preds))
                            top_conf = float(preds[top_idx])
                            # percentages dict
                            classes = {k: v for k, v in labels.items()}
                            prob_dict = {}
                            for idx_str, cls_name in classes.items():
                                # idx_str may be "0" or "1"; ensure numeric idx
                                idx = int(idx_str)
                                prob_dict[cls_name] = float(preds[idx]) * 100.0

                            # If top_conf below threshold, mark as invalid
                            if top_conf < CONFIDENCE_THRESHOLD:
                                st.warning(
                                    f"Hasil Prediksi: **Gambar Tidak Valid atau Tidak Relevan** — confidence tertinggi {top_conf*100:.2f}%.\n"
                                    "Silakan unggah gambar mata yang fokus, terang, dan jelas."
                                )
                                st.json({k: f"{v:.2f}%" for k, v in prob_dict.items()})
                            else:
                                predicted_label = classes.get(str(top_idx), "unknown")
                                # show probabilities nicely
                                st.write("### Hasil Prediksi")
                                if predicted_label == "normal":
                                    st.success(f"Mata terdeteksi: **NORMAL** — confidence {top_conf*100:.2f}%")
                                elif predicted_label == "cataract":
                                    st.error(f"Mata terdeteksi: **INDIKASI KATARAK** — confidence {top_conf*100:.2f}%")
                                else:
                                    st.info(f"Hasil: {predicted_label} — confidence {top_conf*100:.2f}%")

                                # show both class percentages
                                st.write("#### Probabilitas per kelas:")
                                for cls, pct in prob_dict.items():
                                    st.write(f"- **{cls}**: {pct:.2f}%")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat prediksi: {e}")

with col2:
    st.markdown("## Info & Petunjuk")
    st.write(
        """
        - Pastikan foto fokus pada mata (tutup rambut/kacamata yang menghalangi).
        - Hindari gambar buram atau sangat gelap/terang.
        - Hasil hanya sebagai indikasi; **konsultasikan dokter mata** untuk diagnosis definitif.
        """
    )
    st.markdown("## Debug (informasi model)")
    if model is None:
        st.markdown("- Model: **Tidak tersedia**")
    else:
        st.markdown(f"- Model source: **{model_source}**")
        st.markdown(f"- Model summary (ringkas):")
        # print a short model summary (catch any error if the model is complex)
        try:
            summary_str = []
            model.summary(print_fn=lambda s: summary_str.append(s))
            st.text("\n".join(summary_str[:30]))  # show first ~30 lines to avoid long output
        except Exception:
            st.text("summary not available")

# Footer: show label mapping
st.markdown("---")
st.write("Label mapping:", labels)

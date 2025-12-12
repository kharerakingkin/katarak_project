import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable

# ==============================================================
# REGISTER CUSTOM TRANSFORMER LAYER (HARUS ADA)
# ==============================================================


@register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, hidden_dim=256, ff_dim=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout
        )

        self.ffn_dense1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_dense2 = layers.Dense(hidden_dim)

        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn_dense2(self.ffn_dense1(x))
        x = self.norm2(x + ffn_out)

        return x


# ==============================================================
# LOAD MODEL (.keras)
# ==============================================================

MODEL_PATH = "models/cataract_vit_model.keras"
model = None

try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"TransformerBlock": TransformerBlock}
    )
    print("Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()


# ==============================================================
# STREAMLIT UI SETTINGS
# ==============================================================

st.set_page_config(
    page_title="Cataract Detection",
    page_icon="👁",
    layout="centered"
)

st.markdown("""
<style>
body { background-color: #f5f7fa; }
.upload-box {
    border: 2px dashed #bbb;
    padding: 20px;
    border-radius: 12px;
    background: #fafafa;
    text-align: center;
}
.prediction-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# ==============================================================
# IMAGE PREPROCESSING
# ==============================================================

IMG_SIZE = (224, 224)
LABELS = {0: "cataract", 1: "normal"}


def preprocess(image: Image.Image):
    img = image.resize(IMG_SIZE)
    img = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict(image: Image.Image):
    arr = preprocess(image)
    preds = model.predict(arr)[0]
    return preds


# ==============================================================
# PAGE TITLE
# ==============================================================

st.title("👁 Cataract Detection (MobileNetV3 + ViT Tail)")
st.write("Unggah gambar mata untuk dianalisis menggunakan model Anda.")


# ==============================================================
# FILE UPLOADER
# ==============================================================

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)


# ==============================================================
# RUN PREDICTION
# ==============================================================

if uploaded_file and st.button("🔍 Analisis"):
    with st.spinner("Menganalisis gambar..."):
        preds = predict(img)

        class_id = int(np.argmax(preds))
        confidence = float(preds[class_id]) * 100

        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.subheader("Hasil Prediksi")
        st.write(f"**Label:** {LABELS[class_id].upper()}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))
        st.markdown("</div>", unsafe_allow_html=True)

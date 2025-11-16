import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import streamlit as st

# ==========================
# KONFIGURASI
# ==========================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_final.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ==========================
# REGISTER CUSTOM LAYER
# ==========================
try:
    register_serializable = keras.saving.register_keras_serializable
except:
    register_serializable = keras.utils.register_keras_serializable

@register_serializable(package="Custom")
class PositionEmbedding(layers.Layer):
    """Harus ada, karena disimpan dalam file .keras hasil training."""
    def build(self, input_shape):
        seq_len = input_shape[1]
        dim = input_shape[2]
        self.pos_emb = self.add_weight(
            shape=(seq_len, dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_emb


@register_serializable(package="Custom")
class TransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, ff_dim=128, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def build(self, input_shape):
        embed_dim = input_shape[-1]

        # Hindari error key_dim=0
        key_dim = max(1, embed_dim // self.num_heads)

        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.dropout)
        self.dropout2 = layers.Dropout(self.dropout)
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
    try:
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,
            custom_objects={
                "TransformerBlock": TransformerBlock,
                "PositionEmbedding": PositionEmbedding
            }
        )
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model:\n{e}")
        st.stop()

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)

model = load_model()
labels = load_labels()

# ==========================
# FUNGSI PREDIKSI
# ==========================
def predict(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    return preds


# ==========================
# STREAMLIT UI
# ==========================
st.title("👁️ Deteksi Katarak Berbasis AI")

uploaded_file = st.file_uploader("📤 Upload foto mata", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diunggah", width=300)

    if st.button("🔍 Deteksi"):
        with st.spinner("Menganalisis gambar..."):
            preds = predict(image)

            cataract_prob = preds[0] * 100
            normal_prob = preds[1] * 100
            confidence = np.max(preds) * 100

            classes = list(labels.values())
            predicted = classes[np.argmax(preds)]

            if confidence < 70:
                predicted = "irrelevant"

        st.subheader("Hasil Deteksi")

        if predicted == "cataract":
            st.error(f"⚠️ Katarak terdeteksi ({cataract_prob:.2f}%)")
        elif predicted == "normal":
            st.success(f"✅ Mata Normal ({normal_prob:.2f}%)")
        else:
            st.warning("❓ Gambar tidak relevan / bukan mata")

        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"Cataract: {cataract_prob:.2f}%")
        st.write(f"Normal: {normal_prob:.2f}%")

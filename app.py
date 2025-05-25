import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

st.title("🍨 Ice Cream vs 🧇 Waffles Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])

@st.cache_resource
def load_cnn_model():
    return load_model("ice_vs_waffles_cnn_1.h5")
model = load_cnn_model()
class_names = ['Ice Cream', 'Waffles']

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = class_names[int(prediction[0] > 0.5)]
    confidence = float(prediction[0]) if prediction[0] > 0.5 else 1 - float(prediction[0])

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### It looks like : `{predicted_class}`.")
    st.markdown(f"### And I'm : `{confidence:.2f}%`sure of it.")
    st.markdown(f"📊 Raw Output: `{prediction}`")

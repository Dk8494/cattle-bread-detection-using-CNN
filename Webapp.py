import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import matplotlib.pyplot as plt

# ---------------------------
# Load model and class names
# ---------------------------
model = load_model("/Users/devendrakumar/Coding /Python Lib and basic/SIH/cattle_model1.h5")
DATASET_DIR = r"/Users/devendrakumar/Coding /Python Lib and basic/SIH/Cattle Breeds"
class_names = sorted(os.listdir(DATASET_DIR))

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄", layout="wide")

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🐄 Cattle Breed Classification</h1>
    <p style='text-align: center;'>Upload a cattle image and let the AI predict its breed with confidence!</p>
    """,
    unsafe_allow_html=True,
)

import streamlit as st

# Set page title
st.set_page_config(page_title="Cattle Breeds Classifier", layout="centered")

import base64

def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    body {{
    background-image: url("data:image/png;base64,{encoded}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    .stApp {{
    background-color: rgba(0,0,0,0.7);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_local("/Users/devendrakumar/Coding /Python Lib and basic/SIH/61Yg-SguuBL.jpg")


st.title("Cattle Breeds Classification")
st.write("Upload an image to classify the breed of cattle.")


st.sidebar.title("📌 About")
st.sidebar.info(
    "This AI tool identifies **cattle and buffalo breeds** using deep learning.\n\n"
    "- Powered by TensorFlow/Keras\n"
    "- Built for SIH 2025\n"
    "- Supports integration with Bharat Pashudhan App"
)
st.sidebar.header("📂 Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Preprocess image
        img_height, img_width = 180, 180
        img = image.resize((img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.subheader("🔮 Prediction Result")
        st.success(f"✅ **{predicted_class}**")
        st.progress(int(confidence))

        # Show all class probabilities as a bar chart
        st.subheader("📊 Confidence by Breed")
        prob_dict = {class_names[i]: float(score[i]) for i in range(len(score))}

        # Example probabilities
        breeds = list(prob_dict.keys())
        probs = list(prob_dict.values())

        plt.figure(figsize=(8, 4))
        plt.bar(breeds, probs,color='black')  # <-- Set colors here
        plt.ylabel("Probability")
        plt.title("Cattle Breed Prediction")
        st.pyplot(plt)

else:
    st.info("👈 Upload an image from the sidebar to get started!")

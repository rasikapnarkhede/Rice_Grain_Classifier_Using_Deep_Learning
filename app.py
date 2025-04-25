import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
model = load_model('Python/Rice_Classifier/rice_cnn_model.h5')  # path to your saved model
class_names = ['Basmati', 'Arborio', 'Ipsala', 'Jasmine', 'Karacadag']  # Update as per your dataset

st.title("🍚 Rice Grain Classifier")
st.subheader("🌾 Identify the variety of rice grain with deep learning")
st.write("Upload a rice grain image to predict its type.")


uploaded_file = st.file_uploader("Choose a rice image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((28, 28))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]



    st.success(f"Predicted Rice Type: *{pred_class}*")
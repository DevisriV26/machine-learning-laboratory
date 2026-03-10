import streamlit as st
import numpy as np
from PIL import Image
import pickle

# Load trained model
mlp_model = pickle.load(open("mlp_emnist_model.pkl", "rb"))

st.title("Handwritten Letter Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Convert image to grayscale
    image = Image.open(uploaded_file).convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image_array = np.array(image)

    # Invert colors if needed
    if np.mean(image_array) > 127:
        image_array = 255 - image_array

    # Normalize
    image_array = image_array / 255.0

    # Flatten
    image_array = image_array.reshape(1, -1)

    # Predict
    prediction = mlp_model.predict(image_array)[0]

    # Convert number to letter
    predicted_letter = chr(prediction + 64)

    st.write("Predicted Character:", predicted_letter)
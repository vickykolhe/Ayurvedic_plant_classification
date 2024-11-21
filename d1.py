import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("resnet_mod.h5")  # Update this to your actual model filename

# Define class names corresponding to the model's output
class_names = [
    "Nooni",
    "Nithyapushpa",
    "Basale",
    "Pomegranate",
    "Honge",
    "Lemon_grass",
    "Mint",
    "Betel_Nut",
    "Nagadali",
    "Curry_Leaf",
    "Jasmine",
    "Castor",
    "Sapota",
    "Neem",
    "Ashoka",
    "Brahmi",
    "Amruta_Balli",
    "Pappaya",
    "Pepper",
    "Wood_sorel",
    "Gauva",
    "Hibiscus",
    "Ashwagandha",
    "Aloevera",
    "Raktachandini",
    "Insulin",
    "Bamboo",
    "Amla",
    "Arali",
    "Geranium",
    "Avacado",
    "Lemon",
    "Ekka",
    "Betel",
    "Henna",
    "Doddapatre",
    "Rose",
    "Mango",
    "Tulasi",
    "Ganike",
]

# Streamlit app title with styled markdown
st.markdown(
    """
    <style>
    .main-title {
        font-size:36px; 
        color:#4CAF50; 
        font-weight: bold;
        text-align: center;
    }
    .sub-title {
        font-size:20px; 
        color:#555;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 40px;
    }
    </style>
    <h1 class="main-title">🌿Ayurvedic Plant Classification App🌿</h1>
    <h2 class="sub-title">Identify plants with AI-powered predictions</h2>
    """,
    unsafe_allow_html=True,
)

# Upload image file
uploaded_file = st.file_uploader(
    "Upload an image of the plant:", type=["jpg", "jpeg", "png"]
)

# Display columns to manage layout
if uploaded_file is not None:
    # Layout split into two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True, channels="RGB")

    with col2:
        # Preprocess the image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (224, 224))  # Resize to the input size of your model
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Loading animation
        with st.spinner("Predicting..."):
            # Make a prediction
            predictions = model.predict(image)

        # Get the index of the class with the highest probability
        predicted_index = np.argmax(predictions)

        # Get the corresponding class name
        predicted_class_name = class_names[predicted_index]

        # Display the prediction result with some styling
        st.markdown(
            f"""
            <div style='text-align:center'>
                <h2>Prediction Result</h2>
                <h1>{predicted_class_name}</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Add a stylish footer to the app
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer-text {
        font-size: 12px;
        text-align: center;
        color: #888;
        margin-top: 30px;
    }
    </style>
    <div class="footer-text">
    </div>
    """,
    unsafe_allow_html=True,
)

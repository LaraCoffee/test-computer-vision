import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env



# --- Hugging Face API ---
# API_TOKEN = st.secrets["HF_Token"]
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

API_TOKEN = os.getenv("HF_Token")
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    # We set Content-Type dynamically based on image format below
}

# --- Streamlit UI ---
st.title("🧠 Image Classification (ViT - Hugging Face API)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.read()

    # Display image
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Set correct content type based on image format
    mime_type = uploaded_file.type  # This gives 'image/jpeg', 'image/png', etc.
    HEADERS["Content-Type"] = mime_type

    # Send raw image bytes directly
    with st.spinner("Classifying image..."):
        response = requests.post(API_URL, headers=HEADERS, data=image_bytes)

    # Handle response
    if response.status_code == 200:
        predictions = response.json()
        results = "\n".join([f"{p['label']} ({p['score']:.2%})" for p in predictions[:3]])
        st.text_area("Predictions", value=results, height=100)
    else:
        st.error(f"API Error {response.status_code}: {response.text}")

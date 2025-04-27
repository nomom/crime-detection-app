import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import shutil
import zipfile
import io
import gdown

# Download best.pt if not present
if not os.path.exists("best.pt"):
    gdown.download("https://drive.google.com/uc?id=abc123xyz", "best.pt", quiet=False)

# Set page configuration
st.set_page_config(page_title="Crime Detection App", layout="wide")

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.results = []

# Create temporary directories
TEMP_DIR = "temp"
OUTPUT_DIR = "output"
for dir in [TEMP_DIR, OUTPUT_DIR]:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

yolo_model = load_yolo_model()

# Streamlit UI
st.title("Crime Detection App")
st.write("Upload images for object detection using YOLOv11.")
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files and st.button("Process Images"):
    st.session_state.results = []
    for uploaded_file in uploaded_files:
        # Save uploaded image
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load image
        image = Image.open(file_path)
        image_np = np.array(image)

        # Perform detection
        results = yolo_model(image_np, conf=confidence_threshold)
        detected_image = results[0].plot()

        # Convert to PIL for display
        detected_image_pil = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

        # Save detected image
        output_path = os.path.join(OUTPUT_DIR, f"detected_{uploaded_file.name}")
        detected_image_pil.save(output_path)

        # Store results
        st.session_state.results.append({
            "original": image,
            "detected": detected_image_pil,
            "filename": uploaded_file.name
        })

    st.session_state.processed = True

# Display results
if st.session_state.processed and st.session_state.results:
    for result in st.session_state.results:
        col1, col2 = st.columns(2)
        with col1:
            st.image(result["original"], caption="Original Image", use_column_width=True)
        with col2:
            st.image(result["detected"], caption="Detected Image", use_column_width=True)

    # Download zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for result in st.session_state.results:
            detected_path = os.path.join(OUTPUT_DIR, f"detected_{result['filename']}")
            zip_file.write(detected_path, f"detected_{result['filename']}")
    zip_buffer.seek(0)
    st.download_button(
        label="Download Detected Images",
        data=zip_buffer,
        file_name="detected_images.zip",
        mime="application/zip"
    )

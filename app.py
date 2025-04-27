
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from realesrgan import RealESRGAN
import shutil
import zipfile
import io
import gdown

# Download best.pt from Google Drive if not present
if not os.path.exists("best.pt"):
    gdown.download("YOUR_GOOGLE_DRIVE_LINK", "best.pt", quiet=False)

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

# Load models
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")  # Trained YOLOv11 model
    esrgan_model = RealESRGAN("RealESRGAN_x4plus")  # Pre-trained Real-ESRGAN
    return yolo_model, esrgan_model

yolo_model, esrgan_model = load_models()

# Function to upscale image with Real-ESRGAN
def upscale_image(image, outscale=3.5):
    img = np.array(image)
    upscaled_img = esrgan_model.enhance(img, outscale=outscale)[0]
    return Image.fromarray(upscaled_img)

# Function to run YOLOv11 inference
def run_yolo_inference(image, model, confidence=0.5):
    results = model.predict(image, conf=confidence)
    result_img = results[0].plot()  # Image with bounding boxes
    return Image.fromarray(result_img), results[0]

# Main app
st.title("Crime Scene Object Detection App")
st.markdown(
    """
    Upload crime scene images to detect objects using a trained YOLOv11 model.
    - **Optional Enhancement**: Upscale images with Real-ESRGAN (3.5x) for improved clarity.
    - **Detection**: Identify objects (e.g., weapons, suspects) with adjustable confidence.
    - **Results**: View original, upscaled (if enabled), and detected images with bounding boxes.
    """
)

# Sidebar for options
st.sidebar.header("Detection Options")
use_esrgan = st.sidebar.checkbox("Enable Real-ESRGAN Upscaling", value=False)
confidence_threshold = st.sidebar.slider(
    "YOLOv11 Confidence Threshold", 0.1, 0.9, 0.5, 0.05
)

# File uploader
uploaded_files = st.file_uploader(
    "Upload Crime Scene Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state.results = []
    st.session_state.processed = False

    # Process each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        # Save uploaded image
        image_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Initialize result dictionary
        result = {"filename": uploaded_file.name, "original": image}
        
        # Upscale if enabled
        if use_esrgan:
            with st.spinner(f"Upscaling {uploaded_file.name}..."):
                upscaled_image = upscale_image(image)
                result["upscaled"] = upscaled_image
                upscaled_path = os.path.join(OUTPUT_DIR, f"upscaled_{uploaded_file.name}")
                upscaled_image.save(upscaled_path)
        else:
            result["upscaled"] = None
        
        # Run YOLO inference
        with st.spinner(f"Detecting objects in {uploaded_file.name}..."):
            inference_image = result["upscaled"] if use_esrgan else image
            result_image, yolo_result = run_yolo_inference(
                inference_image, yolo_model, confidence_threshold
            )
            result["detection"] = result_image
            result["yolo_result"] = yolo_result
            
            # Save detection result
            detection_path = os.path.join(OUTPUT_DIR, f"detected_{uploaded_file.name}")
            result_image.save(detection_path)
        
        st.session_state.results.append(result)
    
    st.session_state.processed = True

# Display results
if st.session_state.processed and st.session_state.results:
    st.header("Detection Results")
    for result in st.session_state.results:
        st.subheader(f"Image: {result['filename']}")
        cols = st.columns(3 if use_esrgan else 2)
        
        # Original image
        with cols[0]:
            st.image(result["original"], caption="Original Image", use_column_width=True)
        
        # Upscaled image
        if use_esrgan:
            with cols[1]:
                st.image(result["upscaled"], caption="Upscaled Image (Real-ESRGAN)", use_column_width=True)
        
        # Detection result
        with cols[-1]:
            st.image(result["detection"], caption="YOLOv11 Detection", use_column_width=True)
            
            # Display detection details
            boxes = result["yolo_result"].boxes
            if boxes:
                st.write("Detected Objects:")
                for box in boxes:
                    class_name = result["yolo_result"].names[int(box.cls)]
                    conf = box.conf.item()
                    st.write(f"- {class_name}: Confidence {conf:.2f}")
            else:
                st.write("No objects detected.")
    
    # Download results as zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in os.listdir(OUTPUT_DIR):
            file_path = os.path.join(OUTPUT_DIR, file)
            zip_file.write(file_path, file)
    zip_buffer.seek(0)
    st.download_button(
        label="Download Processed Images",
        data=zip_buffer,
        file_name="detected_images.zip",
        mime="application/zip",
    )

# Clean up
if st.button("Clear Results"):
    st.session_state.processed = False
    st.session_state.results = []
    for dir in [TEMP_DIR, OUTPUT_DIR]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    st.success("Results cleared.")

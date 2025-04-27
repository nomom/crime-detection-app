import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import gdown
from basicsr.archs.rrdbnet_arch import RRDBNet
import shutil
import zipfile
import io

# Download best.pt and RealESRGAN_x4plus.pth
if not os.path.exists("best.pt"):
    gdown.download("YOUR_BEST_PT_GOOGLE_DRIVE_LINK", "best.pt", quiet=False)
if not os.path.exists("weights/RealESRGAN_x4plus.pth"):
    os.makedirs("weights", exist_ok=True)
    gdown.download("YOUR_REALESRGAN_PTH_GOOGLE_DRIVE_LINK", "weights/RealESRGAN_x4plus.pth", quiet=False)

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
    yolo_model = YOLO("best.pt")
    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    esrgan_model.load_state_dict(torch.load("weights/RealESRGAN_x4plus.pth")['params_ema'])
    esrgan_model.eval()
    esrgan_model = esrgan_model.to(torch.device("cpu"))
    return yolo_model, esrgan_model

yolo_model, esrgan_model = load_models()

# Upscale image
def upscale_image(image, outscale=3.5):
    img = np.array(image)
    img = img[:, :, ::-1].copy()  # RGB to BGR
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        output = esrgan_model(img_tensor)
    output = output.squeeze().permute(1, 2, 0).clamp(0, 1) * 255.0
    output = output.numpy().astype(np.uint8)
    output = output[:, :, ::-1]  # BGR to RGB
    output = cv2.resize(output, None, fx=outscale/4, fy=outscale/4, interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(output)

# Streamlit UI
st.title("Crime Detection App")
st.write("Upload images for object detection using YOLOv11 with optional upscaling.")
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
upscale = st.checkbox("Enable Real-ESRGAN Upscaling (3.5x)")
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

        # Upscale if enabled
        if upscale:
            upscaled_image = upscale_image(image)
            upscaled_np = np.array(upscaled_image)
            upscaled_path = os.path.join(OUTPUT_DIR, f"upscaled_{uploaded_file.name}")
            upscaled_image.save(upscaled_path)
        else:
            upscaled_image = None
            upscaled_np = image_np

        # Perform detection
        results = yolo_model(upscaled_np, conf=confidence_threshold)
        detected_image = results[0].plot()
        detected_image_pil = Image.fromarray(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
        output_path = os.path.join(OUTPUT_DIR, f"detected_{uploaded_file.name}")
        detected_image_pil.save(output_path)

        # Store results
        st.session_state.results.append({
            "original": image,
            "upscaled": upscaled_image,
            "detected": detected_image_pil,
            "filename": uploaded_file.name
        })

    st.session_state.processed = True

# Display results
if st.session_state.processed and st.session_state.results:
    for result in st.session_state.results:
        cols = st.columns(3 if result["upscaled"] else 2)
        with cols[0]:
            st.image(result["original"], caption="Original Image", use_column_width=True)
        if result["upscaled"]:
            with cols[1]:
                st.image(result["upscaled"], caption="Upscaled Image", use_column_width=True)
            with cols[2]:
                st.image(result["detected"], caption="Detected Image", use_column_width=True)
        else:
            with cols[1]:
                st.image(result["detected"], caption="Detected Image", use_column_width=True)

    # Download zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for result in st.session_state.results:
            if result["upscaled"]:
                upscaled_path = os.path.join(OUTPUT_DIR, f"upscaled_{result['filename']}")
                zip_file.write(upscaled_path, f"upscaled_{result['filename']}")
            detected_path = os.path.join(OUTPUT_DIR, f"detected_{result['filename']}")
            zip_file.write(detected_path, f"detected_{result['filename']}")
    zip_buffer.seek(0)
    st.download_button(
        label="Download Detected Images",
        data=zip_buffer,
        file_name="detected_images.zip",
        mime="application/zip"
    )

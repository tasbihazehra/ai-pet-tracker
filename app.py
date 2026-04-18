import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load a very small, fast model
model = YOLO('yolov8n.pt') 

st.set_page_config(page_title="AI Pet Detector", page_icon="🐾")
st.title("🐾 AI Pet Detection & Tracking")

# Simple instruction
st.write("Upload a photo of a pet or a street scene to see the AI work!")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Open image
    img = Image.open(uploaded_file)
    
    # Run the AI
    results = model.track(img, persist=True)
    
    # Draw the results on the image
    res_plotted = results[0].plot()
    
    # Display the final image
    st.image(res_plotted, caption="AI Detection results", use_container_width=True)
    
    st.success("Detection Successful!")

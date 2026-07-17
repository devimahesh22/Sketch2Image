import streamlit as st
import pandas
from PIL import Image
from preprocessing.preprocess import resize_image, greyscale_image, remove_noise, threshold_image, detect_edges, morphological_operation, draw_contours, extract_green_mask
import numpy as np
import cv2


st.title("Sketch2Circuit AI")


uploaded_image = st.file_uploader("Add a file")

col1, col2 = st.columns(2)

#Display image
if uploaded_image is not None:
    with col1:
        image_uploaded = st.image(uploaded_image)
        pil_image = Image.open(uploaded_image)
        st.caption("Uploaded Image")
        image = np.array(pil_image)
    with col2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = resize_image(image)
        st.image(image, channels="BGR")
        st.caption("Resized Image")
    with col1:
        image = greyscale_image(image)
        st.image(image)
        st.caption("Greyscaled Image")
    with col2:
        image = remove_noise(image)
        st.image(image)
        st.caption("Image with noise removed")
    with col1:
        image = threshold_image(image)
        st.image(image)
        st.caption("Threshold Image calculating unique threshold for each pixel depending upon the neighbour (Effective for images with varying illumination)")
    lower_threshold = st.slider("lower_threshold for detecting edge",0,255,100)
    higher_threshold = st.slider("higher_threshold for detecting edge", 0,255,200)
    image = detect_edges(image, lower_threshold, higher_threshold)
    st.image(image)
    image = morphological_operation(image)
    st.image(image)
    image, count = draw_contours(image)
    st.write(f"Contours: {count}")
    st.image(image)


import streamlit as st
import pandas
from PIL import Image
from preprocessing.preprocess import resize_image, greyscale_image, remove_noise, threshold_image, detect_edges, morphological_operation, draw_contours, extract_green_mask
import numpy as np
import cv2
from controlnet.model import load_pipeline
from controlnet.inference import generate_image


st.title("Sketch2Circuit AI")

@st.cache_resource
def get_pipeline():
    return load_pipeline()

pipe = get_pipeline()

uploaded_image = st.file_uploader("Add a file")

col1, col2 = st.columns(2)

#Display image
if uploaded_image is not None:
    with col1:
        image_uploaded = st.image(uploaded_image, caption="Uploaded Image")
        pil_image = Image.open(uploaded_image)
        st.image(pil_image, caption = "PIL Image")
        image = np.array(pil_image)
    with col2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = resize_image(image)
        st.image(image, channels="BGR", caption="Resized Image")
    with col1:
        image = greyscale_image(image)
        st.image(image, caption= "Greyscaled Image")
    with col2:
        image = remove_noise(image)
        st.image(image, caption="Image with noise removed")
    with col1:
        threshold = threshold_image(image)
        st.image(threshold, caption= "Threshold Image calculating unique threshold for each pixel depending upon the neighbour (Effective for images with varying illumination)")
    lower_threshold = st.slider("lower_threshold for detecting edge",0,255,100)
    higher_threshold = st.slider("higher_threshold for detecting edge", 0,255,200)
    edges = detect_edges(threshold, lower_threshold, higher_threshold)
    st.image(edges, caption= "Detect Edges")
    morph = morphological_operation(edges)
    st.image(morph, caption="Morphed Image")
    contours, count = draw_contours(morph)
    st.write(f"Contours: {count}")
    st.image(contours, caption = "Contour Image")

    prompt = """
    A clean electrical circuit schematic.
    Technical engineering drawing.
    Preserve every wire and every electronic component.
    Use standard electrical symbols.
    Black lines on a pure white background.
    Flat 2D schematic.
    No perspective.
    No shading.
    No textures.
    No artistic style.
    No extra components.
    """

    negative_prompt = """
    photo,
    realistic,
    painting,
    3d,
    shadow,
    texture,
    noise,
    extra wires,
    extra components,
    extra circles,
    distorted,
    blurry,
    low quality
    """

    control_type = st.selectbox(
    "Control Image",
    ["Threshold", "Edge", "Contour"])

    if control_type == "Threshold":
        control = threshold
    elif control_type == "Edge":
        control = edges
    else:
        control = contours

    if st.button("Generate"):
        generated = generate_image(
            pipe,
            prompt,
            negative_prompt,
            control
        )
        st.success("Generation Complete!")
        st.image(generated, caption="Generated Circuit")
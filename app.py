pip install streamlit opencv-python-headless pillow

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Function to compress image
def compress_image(image, k):
    # Convert image to RGB (OpenCV loads images in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    # Convert to float32 for kmeans
    pixels = np.float32(pixels)

    # Define criteria, number of clusters (K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Get the compressed image
    compressed_image = palette[labels.flatten()]
    compressed_image = compressed_image.reshape(image.shape)
    # Convert to uint8
    compressed_image = np.uint8(compressed_image)
    return compressed_image

# Streamlit UI
st.title('Color Image Compression')
st.write('Upload an image and choose the number of colors (K) to compress the image.')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Slider for selecting K value
    k = st.slider('Select number of colors (K)', min_value=2, max_value=64, value=16)
    
    # Compress image button
    if st.button('Compress Image'):
        compressed_image = compress_image(image, k)
        st.image(compressed_image, caption='Compressed Image', use_column_width=True)
        
        # Option to download the compressed image
        im_pil = Image.fromarray(compressed_image)
        buf = BytesIO()
        im_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Compressed Image", data=byte_im, file_name="compressed_image.jpg", mime="image/jpeg")

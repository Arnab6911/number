import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------
# 1. SET UP THE APP INTERFACE
# -------------------------------------------------
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide")

# -------------------------------------------------
# 2. LOAD THE "BRAIN"
# -------------------------------------------------
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    print("--- Loading model... ---")
    model = tf.keras.models.load_model('svhn_model.h5')
    print("--- Model loaded! ---")
    return model

model = load_my_model()

# -------------------------------------------------
# 3. HELPER FUNCTION TO PROCESS THE IMAGE
# -------------------------------------------------
# We create this function so both tabs can use the same logic
def process_and_predict(image_file):
    # 1. Load the image
    image = Image.open(image_file)
    # Convert from PIL to a NumPy array (OpenCV format)
    img_array = np.array(image)
    # Ensure it's in BGR format if it's color (OpenCV default)
    if img_array.ndim == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Show the image you uploaded/took
    st.image(image, caption="Your Image", width=300)
    
    # 2. Start prediction automatically
    with st.spinner("🧠 Thinking..."):
        
        # 3. Pre-process the image
        # (This is the *exact* same way we processed the training data)
        roi_resized = cv2.resize(img_array, (32, 32))
        roi_normalized = roi_resized.astype('float32') / 255.0
        roi_batch = np.expand_dims(roi_normalized, axis=0)

        # 4. Make the prediction
        prediction = model.predict(roi_batch)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        # 5. Show the result!
        st.success(f"## I see the digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")


# -------------------------------------------------
# 4. DISPLAY THE APP UI
# -------------------------------------------------
st.title("🚀 SVHN Digit Recognizer")
st.write("This app uses your **SVHN (Photograph Expert) CNN model**.")
st.write("Upload a photo or take a new one with your camera.")

# Create the two tabs
tab1, tab2 = st.tabs(["📁 Upload a Photo", "📸 Take a Photo"])

# --- Tab 1: File Uploader ---
with tab1:
    st.header("Upload from your files")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    
    if uploaded_file is not None:
        # Call our helper function
        process_and_predict(uploaded_file)

# --- Tab 2: Camera Input ---
with tab2:
    st.header("Use your camera")
    camera_photo = st.camera_input("Take a picture of a single digit", key="camera")
    
    if camera_photo is not None:
        # Call our helper function
        process_and_predict(camera_photo)
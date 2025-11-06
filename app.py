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
# 3. ADVANCED IMAGE PROCESSING FUNCTION
# -------------------------------------------------
def process_and_predict(image_file):
    # 1. Load the image
    image = Image.open(image_file)
    # Convert from PIL to a NumPy array (OpenCV format)
    img_array = np.array(image)
    
    # 2. --- NEW OPENCV PREPROCESSING ---
    # Convert to color (OpenCV expects BGR)
    if img_array.ndim == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        # If it's grayscale, convert it to BGR
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    # Convert to grayscale for processing
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply a blur to reduce noise, then use Otsu's thresholding
    # This creates a clean black-and-white image of the digit
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (the outlines of the white shapes)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        st.error("I couldn't find a digit. Please try a clearer photo with better contrast.")
        return

    # Find the largest contour, which we assume is our digit
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the "bounding box" (x, y, width, height) of the digit
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 3. --- CROP THE DIGIT ---
    # Crop the *original color image* using the box we found
    # We add a 10-pixel "padding" to make sure we get the whole digit
    padding = 10
    roi_x1 = max(0, x - padding)
    roi_y1 = max(0, y - padding)
    roi_x2 = min(img_bgr.shape[1], x + w + padding)
    roi_y2 = min(img_bgr.shape[0], y + h + padding)
    
    cropped_digit = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if cropped_digit.size == 0:
        st.error("I found something, but the crop failed. Please try again.")
        return

    # 4. --- PREPARE FOR MODEL ---
    # Now we just resize our *perfect crop* to 32x32
    final_image = cv2.resize(cropped_digit, (32, 32))
    
    # Show the user what the model is "seeing"
    st.image(final_image, caption="What the Model Sees (32x32 Processed)", width=150)
    
    # Normalize and add batch dimension
    img_normalized = final_image.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # 5. --- PREDICT ---
    with st.spinner("🧠 Thinking..."):
        prediction = model.predict(img_batch)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        # 6. Show the result!
        st.success(f"## I see the digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")


# -------------------------------------------------
# 4. DISPLAY THE APP UI
# -------------------------------------------------
st.title("🚀 SVHN Smart Digit Recognizer")
st.write("This app now uses **OpenCV** to find the digit in your photo *before* predicting.")
st.info("**Tip:** For best results, use a clear photo with good contrast (e.g., dark number on a light background).")

# Create the two tabs
tab1, tab2 = st.tabs(["📁 Upload a Photo", "📸 Take a Photo"])

# --- Tab 1: File Uploader ---
with tab1:
    st.header("Upload from your files")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")
    
    if uploaded_file is not None:
        process_and_predict(uploaded_file)

# --- Tab 2: Camera Input ---
with tab2:
    st.header("Use your camera")
    camera_photo = st.camera_input("Take a picture of a single digit", key="camera")
    
    if camera_photo is not None:
        process_and_predict(camera_photo)
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------------------------
# 1. SET UP THE APP INTERFACE
# -------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------
# 2. LOAD THE "BRAIN"
# -------------------------------------------------
@st.cache_resource
def load_my_model():
    print("--- Loading model... ---")
    model = tf.keras.models.load_model('svhn_model.h5')
    print("--- Model loaded! ---")
    return model

model = load_my_model()

# -------------------------------------------------
# 3. HELPER FUNCTION: "LETTERBOXING"
# -------------------------------------------------
# This is our fix for "squizzing".
def letterbox_image(image, target_size=(32, 32)):
    img_h, img_w, _ = image.shape
    target_h, target_w = target_size
    scale = min(target_w / img_w, target_h / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
    x_center = (target_w - new_w) // 2
    y_center = (target_h - new_h) // 2
    canvas[y_center:y_center + new_h, x_center:x_center + new_w] = resized_img
    return canvas

# -------------------------------------------------
# 4. ADVANCED IMAGE PROCESSING FUNCTION
# -------------------------------------------------
def process_and_predict(image_file):
    # 1. Load the image
    image = Image.open(image_file)
    img_array = np.array(image)
    
    # 2. --- NEW OPENCV PREPROCESSING ---
    if img_array.ndim == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        st.error("I couldn't find a digit. Please try a clearer photo with better contrast.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 3. --- CROP THE DIGIT ---
    padding = 20 # Add a 20-pixel padding to be safe
    roi_x1 = max(0, x - padding)
    roi_y1 = max(0, y - padding)
    roi_x2 = min(img_bgr.shape[1], x + w + padding)
    roi_y2 = min(img_bgr.shape[0], y + h + padding)
    
    cropped_digit = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if cropped_digit.size == 0:
        st.error("I found something, but the crop failed. Please try again.")
        return

    # 4. --- PREPARE FOR MODEL (NO SQUIZZING!) ---
    # We use our letterbox function on the crop
    final_image = letterbox_image(cropped_digit, (32, 32))
    
    # Show the user what the model is "seeing"
    st.image(final_image, caption="What the Model Sees (Processed)", width=150)
    
    img_normalized = final_image.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # 5. --- PREDICT ---
    with st.spinner("🧠 Thinking..."):
        prediction = model.predict(img_batch, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        # 6. Show the result!
        st.success(f"## I see the digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")


# -------------------------------------------------
# 5. DISPLAY THE APP UI
# -------------------------------------------------
st.title("🚀 SVHN Smart Digit Recognizer")
st.write("This app uses OpenCV to find the digit in your photo *before* predicting.")
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
    
    # This is our new "Guide Box"
    st.markdown("""
        <div style="border: 2px solid #00FF00; height: 200px; width: 200px; margin: 10px auto; display: flex; justify-content: center; align-items: center; font-family: sans-serif; color: gray;">
            Try to center your digit inside a box like this
        </div>
        """, unsafe_allow_html=True)
    
    camera_photo = st.camera_input("Take a picture of a single digit", key="camera")
    
    if camera_photo is not None:
        process_and_predict(camera_photo)
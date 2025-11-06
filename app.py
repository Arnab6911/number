import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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
# This is the fix for "squizzing".
# It resizes an image while preserving its aspect ratio by adding black bars.
def letterbox_image(image, target_size=(32, 32)):
    # Get image size
    img_h, img_w, _ = image.shape
    target_h, target_w = target_size

    # Calculate scale
    scale = min(target_w / img_w, target_h / img_h)

    # New (scaled) dimensions
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize image with aspect ratio preservation
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new black "canvas"
    canvas = np.full((target_h, target_w, 3), 0, dtype=np.uint8)

    # Calculate top-left corner to paste the resized image in the center
    x_center = (target_w - new_w) // 2
    y_center = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_center : y_center + new_h, x_center : x_center + new_w] = resized_img

    return canvas

# -------------------------------------------------
# 4. THE LIVE VIDEO PROCESSOR
# -------------------------------------------------
class SVHNVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Define the Region of Interest (ROI) box
        self.box_size = 200 # A 200x200 box
        self.box_color = (0, 255, 0) # Green
        self.box_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def recv(self, frame):
        # Convert the video frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")
        
        # Get frame dimensions
        h, w, _ = img.shape
        
        # Calculate top-left corner of the centered box
        x1 = (w - self.box_size) // 2
        y1 = (h - self.box_size) // 2
        # Calculate bottom-right corner
        x2 = x1 + self.box_size
        y2 = y1 + self.box_size

        # --- PREDICTION LOGIC ---
        try:
            # 1. CROP the image to the box
            roi = img[y1:y2, x1:x2]
            
            # 2. Pre-process the ROI
            #    THIS IS THE NEW, CORRECT WAY (no squizzing!)
            processed_roi = letterbox_image(roi, (32, 32))
            
            # 3. Normalize and add batch dimension
            roi_normalized = processed_roi.astype('float32') / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            # 4. Make the prediction
            prediction = model.predict(roi_batch, verbose=0)
            
            # 5. Get the result
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            
            # 6. Display the result on the frame
            text = f"Prediction: {predicted_digit} ({confidence:.1f}%)"
            cv2.putText(img, text, (x1, y1 - 10), self.font, 0.7, self.box_color, 2)
            
        except Exception as e:
            # If we fail (e.g., at the very start), just skip
            pass
        
        # Draw the ROI box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
        
        # Return the processed frame
        return frame.from_ndarray(img, format="bgr24")

# -------------------------------------------------
# 5. SET UP THE STREAMLIT APP UI
# -------------------------------------------------
st.title("🚀 SVHN Live Digit Detector")
st.info("Center a single digit (e.g., on your phone or paper) inside the green box.")

# Start the webcam feed and apply our video processor
webrtc_streamer(
    key="svhn-detector",
    video_transformer_factory=SVHNVideoTransformer,
    media_stream_constraints={"video": True, "audio": False}, # <--- THIS IS THE FIX
    async_processing=True,
)
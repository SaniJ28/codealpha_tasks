import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model('facial_expression_model.h5')

# Emotion labels - adjust order based on your dataset classes!
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load OpenCV's pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Real-Time Facial Expression Detection")

# Webcam input
img_file_buffer = st.camera_input("Use your webcam to capture video")

if img_file_buffer is not None:
    # Convert to numpy array
    bytes_data = img_file_buffer.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict expression
        preds = model.predict(roi_gray)[0]
        label = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)

        # Draw rectangle and label on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"{label} ({confidence * 100:.2f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show image with detections
    st.image(img, channels="BGR")
else:
    st.write("Please use the webcam above to capture an image.")

#!/usr/bin/env python
# coding: utf-8

# In[24]:


import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pyttsx3
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import librosa
import torch
from models.yolo import Model
from PIL import Image


# In[27]:


# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
  

weights_path = r'C:\Users\sahiy\Downloads\best.pt' 

# Load the YOLOv5 model with the custom weights
model = torch.load(weights_path, map_location=torch.device('cpu'))['model'].float().eval()

def detect_objects(image):
    # Convert image to the required format for the model
    image = Image.fromarray(image)
    image = image.resize((640, 640))  # resize image as needed
    image = np.array(image)
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # Perform inference
    results = model(image)  # Use model(image) depending on the YOLOv5 version

    # Extract bounding boxes, labels, and confidences
    detections = results.xyxy[0].cpu().numpy()  # format: [x1, y1, x2, y2, confidence, class]
    return detections

# Streamlit app
st.title("Street Sign Detection")

uploaded_file = st.file_uploader("Upload an image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    detections = detect_objects(image_np)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detection Results:", detections)

def detect_objects(image_path):
    # Load image
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = np.array(img)

    # Perform inference
    results = model(img)  # or use model(img) depending on the YOLOv5 version

    # Extract bounding boxes, labels, and confidences
    detections = results.xyxy[0].numpy()  # format: [x1, y1, x2, y2, confidence, class]
    
    # Convert results to a more readable format if necessary
    return detections

# Streamlit interface
st.title("📄StreetSmart: Sign & Sound Identifier")
st.write("Upload an image below:")

# File uploader for images
uploaded_image = st.file_uploader("Upload an image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert the image to a format suitable for YOLO
    img_bytes = uploaded_image.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Perform detection
    results = model(img)

    # Display detection results
    st.image(np.squeeze(results.render()), caption='Detected Image', use_column_width=True)
    
    # Extract labels and speak them
    detected_labels = results.pandas().xyxy[0]['name'].tolist()
    if detected_labels:
        labels_text = "Detected: " + ", ".join(detected_labels)
        st.write(labels_text)
        text_to_speech(labels_text)
    else:
        st.write("No signs detected.")
        text_to_speech("No signs detected.")

# File uploader for audio
st.write("Upload an audio file below:")
uploaded_audio = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_audio:
    # Assuming you have a model to detect street sounds from audio files
    # You can integrate it here similar to the image detection process
    st.audio(uploaded_audio, format='audio/wav')
    
    # Dummy placeholder for audio detection - replace with actual model inference
    detected_sound = "traffic sound"  # Replace with actual detection logic
    st.write(f"Detected sound: {detected_sound}")
    text_to_speech(f"Detected sound: {detected_sound}")


# In[28]:


# Streamlit interface
st.title("📄StreetSmart: Sign & Sound Identifier")
st.write("Upload an image below:")

# File uploader for images
uploaded_image = st.file_uploader("Upload an image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Convert the image to a format suitable for YOLO
    img_bytes = uploaded_image.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Perform detection
    results = model(img)

    # Display detection results
    st.image(np.squeeze(results.render()), caption='Detected Image', use_column_width=True)
    
    # Extract labels and speak them
    detected_labels = results.pandas().xyxy[0]['name'].tolist()
    if detected_labels:
        labels_text = "Detected: " + ", ".join(detected_labels)
        st.write(labels_text)
        text_to_speech(labels_text)
    else:
        st.write("No signs detected.")
        text_to_speech("No signs detected.")

# File uploader for audio
st.write("Upload an audio file below:")
uploaded_audio = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_audio:
    # Assuming you have a model to detect street sounds from audio files
    # You can integrate it here similar to the image detection process
    st.audio(uploaded_audio, format='audio/wav')
    
    # Dummy placeholder for audio detection - replace with actual model inference
    detected_sound = "traffic sound"  # Replace with actual detection logic
    st.write(f"Detected sound: {detected_sound}")
    text_to_speech(f"Detected sound: {detected_sound}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from PIL import Image
import sys
import os
from ultralytics import YOLO 
import io
import soundfile as sf
import tempfile


# In[2]:


# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
  
@st.cache_resource
def load_yolo_model():
    try:
        # Load YOLOv5 model
        model = YOLO(r'C:/Users/sahiy/Downloads/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

@st.cache_resource
def load_audio_model():
    try:
        model = tf.keras.models.load_model(r'C:/Users/sahiy/Downloads/audio_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading audio model: {e}")
        return None

yolo_model = load_yolo_model()
audio_model = load_audio_model()

def detect_objects(image):
    results = yolo_model(image)  # Perform inference with YOLO model
    detections = results[0]  # Get the first result (assuming batch size of 1)
    return detections


def preprocess_audio(audio_data, target_shape):
    # Ensure the audio_data is reshaped to match the target shape
    # Flatten the audio_data if it has more than 1 dimension
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # Ensure the length matches the target_shape
    audio_data = audio_data[:target_shape]
    
    # Pad with zeros if necessary
    if len(audio_data) < target_shape:
        audio_data = np.pad(audio_data, (0, target_shape - len(audio_data)))
    
    return np.expand_dims(audio_data, axis=0)  # Add batch dimension


# Streamlit interface
st.title("StreetSmart: Sign and Sound Detection")
st.write("Hello, welcome to StreetSmart! Upload an image or audio file to get started.")

# Upload an image
uploaded_image = st.file_uploader("Upload an image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"], key="image_uploader")

if uploaded_image is not None:
    try:
        # Read the uploaded file as bytes
        img_bytes = uploaded_image.read()
        
        # Open the image using PIL
        img = Image.open(io.BytesIO(img_bytes))
        
        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)
        
        # Convert image to RGB if it's not
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert image to numpy array
        img = np.array(img)
        
        # Detect objects in the image
        detections = detect_objects(img)
        
        # Display results
        prediction = yolo_model.predict(img, show=True)

        prediction_object = prediction[0]
        result = prediction_object.verbose()
        text_to_speech(result[2:])
        original_shape = prediction_object.orig_img.shape

        tensor_shape = prediction_object.boxes.xywh[0]
        x = tensor_shape[0]
        y = tensor_shape[1]
        
        if (x > original_shape[0] * 0.75) and (y < original_shape[1] * 0.25):
            st.write("top right")
            text_to_speech("top right")
        elif (x < original_shape[0] * 0.25) and (y < original_shape[1] * 0.25):
            st.write("top left")
            text_to_speech("top left")
        elif (x > original_shape[0] * 0.75) and (y > original_shape[1] * 0.75):
            st.write("bottom right")
            text_to_speech("bottom right")
        elif (x < original_shape[0] * 0.25) and (y > original_shape[1] * 0.75):
            st.write("bottom left")
            text_to_speech("bottom left")
        else:
            if (x > original_shape[0] * 0.75):
                st.write("To the right")
                text_to_speech("To the right")
            elif (x < original_shape[0] * 0.25):
                st.write("To the left")
                text_to_speech("To the left")
            elif (y < original_shape[1] * 0.25):
                st.write("Above you")
                text_to_speech("Above you")
            elif (y > original_shape[1] * 0.75):
                st.write("Below you")
                text_to_speech("Below you")
            else:
                st.write("in front of you")
                text_to_speech("in front of you")

    except Exception as e:
        st.error(f"Error processing image: {e}")

# Upload an audio file
uploaded_audio = st.file_uploader("Upload an audio file (.wav, .mp3)", type=["wav", "mp3"], key="audio_uploader")

if uploaded_audio is not None:
    try:
        # Save the uploaded audio file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_audio.read())
            temp_file_path = temp_file.name
        
        # Load and process the audio file
        audio_data, samplerate = sf.read(temp_file_path)
        
        # Preprocess audio data to match model input shape
        target_shape = 38  # Adjust based on model requirements
        audio_tensor = preprocess_audio(audio_data, target_shape)
        
        # Perform audio classification (Assuming the audio model is a classifier)
        predictions = audio_model.predict(audio_tensor)  # Add batch dimension
        
        # Display results
        st.write("Audio Predictions:")
        st.write(predictions)
        
        if np.argmax(predictions) == 0:
            st.write("Car Honking")
            text_to_speech("Car Honking")
        if np.argmax(predictions) == 1:
            st.write("Dog Barking")
            text_to_speech("Dog Barking")
        if np.argmax(predictions) == 2:
            st.write("Drilling")
            text_to_speech("Drilling")
        if np.argmax(predictions) == 3:
            st.write("Engine Idling")
            text_to_speech("Engine Idling")
        if np.argmax(predictions) == 4:
            st.write("Jackhammer")
            text_to_speech("Jackhammer")
        if np.argmax(predictions) == 5:
            st.write("Sirens")
            text_to_speech("Sirens")
        else:
            st.write("Violence")
            text_to_speech("Violence")
        
        
    except Exception as e:
        st.error(f"Error processing audio: {e}")


# In[ ]:





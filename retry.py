import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('C://Users//neeli//sign_vision//word_generated_model.keras')

# Define parameters
dataset_folder = 'C://Users//neeli//sign_vision//ISL custom Data'
image_size = (128, 128)

# Initialize ImageDataGenerator to get class labels
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    dataset_folder,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# Function to preprocess an image for model prediction
def preprocess_image(image, image_size=(128, 128)):
    image = cv2.resize(image, image_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to get prediction from the model
def predict_word_from_image(image_array):
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels.get(predicted_class_index, "Unknown")
    return predicted_label

# Function to retrieve an image based on the word input
def get_image_by_word(word, dataset_folder):
    word_folder_path = os.path.join(dataset_folder, word)
    if not os.path.isdir(word_folder_path):
        st.error(f"No folder found for word '{word}' in the dataset.")
        return None
    image_files = [file for file in os.listdir(word_folder_path) if file.lower().endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) == 0:
        st.error(f"No images found in the folder for word '{word}'.")
        return None
    image_path = os.path.join(word_folder_path, image_files[0])
    return load_img(image_path)  # Load at original resolution

# Set page configuration and custom theme colors
st.set_page_config(page_title="Sign Vision", page_icon="🤲", layout="wide")

# Header Section with Video Background
st.video("C://Users//neeli//sign_vision//Sign Language Video.mp4")  # Replace with a URL or local file path for .mp4
st.title("Sign Vision")
st.markdown("This app enables users to recognize and retrieve images for words based on **Indian Sign Language (ISL)** gestures. "
            "ISL is essential for communication among the Deaf and Hard of Hearing communities in India, fostering inclusion and accessibility.")

# Sidebar Section for Extra Information
st.sidebar.image("C://Users//neeli//sign_vision//wally_for project.png", width=150)  # Sidebar image with reduced width
st.sidebar.header("About Indian Sign Language")
st.sidebar.write("Indian Sign Language (ISL) is a visual language that uses hand gestures, facial expressions, and body language. "
                 "It plays a critical role in ensuring effective communication, enabling greater inclusion in society.")

# Main App Input Type Section
st.subheader("Choose your input type:")
input_type = st.selectbox("Options:", ["Capture Image from Webcam", "Upload an Image", "Enter a Word"])

if input_type == "Capture Image from Webcam":
    if st.button("Capture Image from Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not access webcam.")
        else:
            ret, frame = cap.read()
            cap.release()
            if not ret:
                st.error("Error: Failed to capture image from webcam.")
            else:
                st.image(frame, channels="BGR", caption="Captured Image", width=250)  # Set fixed width for smaller display
                processed_image = preprocess_image(frame)
                predicted_label = predict_word_from_image(processed_image)
                st.success(f"Predicted Word: **{predicted_label}**")

elif input_type == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image for word prediction", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)  # Set fixed width for smaller display
        image = image.resize(image_size)
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predicted_label = predict_word_from_image(image_array)
        st.success(f"Predicted Word: **{predicted_label}**")

elif input_type == "Enter a Word":
    word_input = st.text_input("Enter a word to retrieve its image:")
    if word_input:
        retrieved_image = get_image_by_word(word_input, dataset_folder)
        if retrieved_image is not None:
            st.image(retrieved_image, caption=f"Image for word '{word_input}'", width=250)  # Set fixed width for smaller display

# Footer
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

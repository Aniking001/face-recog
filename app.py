import os
import cv2
import face_recognition
import streamlit as st
from PIL import Image

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
            filenames.append(img_path)
    return images, filenames

def get_face_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings

def find_matching_images(input_image_path, folder_path):
    input_image = cv2.imread(input_image_path)
    input_encodings = get_face_encodings(input_image)

    if len(input_encodings) == 0:
        st.warning("No face found in the input image.")
        return []

    input_encoding = input_encodings[0]  # Assuming single face in selfie

    folder_images, filenames = load_images_from_folder(folder_path)
    matching_images = []

    for img, filename in zip(folder_images, filenames):
        img_encodings = get_face_encodings(img)
        for encoding in img_encodings:
            matches = face_recognition.compare_faces([input_encoding], encoding)
            if True in matches:
                matching_images.append(filename)
                break  # Assuming one face match is enough

    return matching_images

# Set up Streamlit app with custom CSS
st.set_page_config(page_title="Face Matching App", page_icon=":smiley:", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0;  /* Set the background color to a light gray */
    }
    .main {
        background-color: #ffffff;  /* Set the main content area background color to white */
        padding: 20px;
    }
    .title {
        color: #4CAF50;
        font-family: 'Arial Black', sans-serif;
    }
    .header, .subheader, .stTextInput, .stButton {
        margin-bottom: 20px;
    }
    .stTextInput input {
        background-color: #e8f0fe;
        border-radius: 8px;
        border: 1px solid #ccc;
        color: blue;  /* Set the input text color to blue */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .image-container img {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>Face Matching App</h1>", unsafe_allow_html=True)

st.header("Upload the Input Selfie")
uploaded_selfie = st.file_uploader("Choose a selfie...", type=["jpg", "jpeg", "png"])

st.header("Provide the Images Folder Path")
images_folder_path = st.text_input("Images Folder Path")

if st.button("Find Matches"):
    if uploaded_selfie and images_folder_path:
        # Save the uploaded selfie to a temporary file
        selfie_temp_path = "temp_selfie.jpg"
        with open(selfie_temp_path, "wb") as f:
            f.write(uploaded_selfie.getbuffer())

        # Find matching images
        matching_images = find_matching_images(selfie_temp_path, images_folder_path)

        # Display matching images
        if matching_images:
            st.success("Matching images found:")
            for img_path in matching_images:
                st.markdown("<div class='image-container'><img src='" + img_path + "'></div>", unsafe_allow_html=True)
        else:
            st.warning("No matching images found.")
    else:
        st.warning("Please provide both the input selfie and images folder path.")

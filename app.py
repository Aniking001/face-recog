import os
import cv2
import face_recognition
import streamlit as st

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

st.title("Face Matching App")

input_selfie_path = st.text_input("Input Selfie Path")
images_folder_path = st.text_input("Images Folder Path")

if st.button("Find Matches"):
    if input_selfie_path and images_folder_path:
        matching_images = find_matching_images(input_selfie_path, images_folder_path)
        if matching_images:
            st.success("Matching images found:")
            for img in matching_images:
                st.image(cv2.imread(img), caption=img)
        else:
            st.warning("No matching images found.")
    else:
        st.warning("Please provide both input selfie path and images folder path.")

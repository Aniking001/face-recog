import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import face_recognition

def load_images_from_folder(folder, limit=None):
    images = []
    labels = []
    count = 0
    for subdir in os.listdir(folder):
        subdir_path = os.path.join(folder, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if limit and count >= limit:
                    break
                img_path = os.path.join(subdir_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(subdir)
                    count += 1
        if limit and count >= limit:
            break
    return images, labels

def get_face_encodings(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings

# Load datasets
train_folder_path = r"C:\machine learning\facescrub-dataset-master\trains"
validate_folder_path = r"C:\machine learning\facescrub-dataset-master\valid"

print("Loading training images...")
train_images, train_labels = load_images_from_folder(train_folder_path, limit=1)
print(f"Loaded {len(train_images)} training images.")

print("Loading validation images...")
validate_images, validate_labels = load_images_from_folder(validate_folder_path, limit=100)
print(f"Loaded {len(validate_images)} validation images.")

# Generate face encoding for the single training image
print("Generating face encoding for the training image...")
train_encoding = None
if train_images:
    train_encodings = get_face_encodings(train_images[0])
    if train_encodings:
        train_encoding = train_encodings[0]
print("Generated face encoding for the training image.")

# Generate face encodings for validation data
print("Generating face encodings for validation data...")
validate_encodings = []
validate_encoding_labels = []
for img, label in tqdm(zip(validate_images, validate_labels), total=len(validate_images)):
    face_encodings = get_face_encodings(img)
    if face_encodings:
        validate_encodings.append(face_encodings[0])
        validate_encoding_labels.append(label)
print(f"Generated {len(validate_encodings)} face encodings for validation data.")

# Predict the labels of the validation set
print("Predicting validation labels...")
y_pred = []
for encoding in validate_encodings:
    distance = np.linalg.norm(train_encoding - encoding)
    if distance < 0.6:  # You can adjust the threshold as needed
        y_pred.append(train_labels[0])
    else:
        y_pred.append("Unknown")

# Filter out "Unknown" predictions
filtered_y_pred = [pred for pred in y_pred if pred != "Unknown"]
filtered_validate_labels = [label for pred, label in zip(y_pred, validate_encoding_labels) if pred != "Unknown"]

# Accuracy
accuracy = accuracy_score(filtered_validate_labels, filtered_y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

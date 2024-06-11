import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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

def augment_image(image):
    # Apply basic augmentations such as flipping and rotation
    augmented_images = [image]
    augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))  # Rotate by 10 degrees
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))  # Rotate by -10 degrees
    return augmented_images

def preprocess_images(images, labels):
    encodings = []
    encoding_labels = []
    for img, label in tqdm(zip(images, labels), total=len(images)):
        augmented_images = augment_image(img)
        for augmented_img in augmented_images:
            face_encodings = get_face_encodings(augmented_img)
            if face_encodings:
                encodings.append(face_encodings[0])
                encoding_labels.append(label)
    return encodings, encoding_labels

# Load datasets
train_folder_path = r"C:\machine learning\facescrub-dataset-master\trains"
validate_folder_path = r"C:\machine learning\facescrub-dataset-master\valid"

print("Loading training images...")
train_images, train_labels = load_images_from_folder(train_folder_path, limit=1000)
print(f"Loaded {len(train_images)} training images.")

print("Loading validation images...")
validate_images, validate_labels = load_images_from_folder(validate_folder_path, limit=100)
print(f"Loaded {len(validate_images)} validation images.")

# Generate face encodings with data augmentation
print("Generating face encodings for training data...")
train_encodings, train_encoding_labels = preprocess_images(train_images, train_labels)
print(f"Generated {len(train_encodings)} face encodings for training data.")

print("Generating face encodings for validation data...")
validate_encodings, validate_encoding_labels = preprocess_images(validate_images, validate_labels)
print("Generating face encodings for validation data...")
validate_encodings, validate_encoding_labels = preprocess_images(validate_images, validate_labels)
print(f"Generated {len(validate_encodings)} face encodings for validation data.")

# Feature scaling
scaler = StandardScaler()
train_encodings = scaler.fit_transform(train_encodings)
validate_encodings = scaler.transform(validate_encodings)

# Train the model using KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_encodings, train_encoding_labels)

# Predict the labels of the validation set
print("Predicting validation labels...")
y_pred = knn.predict(validate_encodings)

# Accuracy
accuracy = accuracy_score(validate_encoding_labels, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


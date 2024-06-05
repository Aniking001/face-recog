import os
import cv2
import face_recognition
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

# Load the FaceScrub training dataset
train_folder_path = r"C:\machine learning\facescrub-dataset-master\train_small"  # Replace with your train dataset path
validate_folder_path = r"C:\machine learning\facescrub-dataset-master\validate_small"  # Replace with your validation dataset path

print("Loading training images...")
train_images, train_labels = load_images_from_folder(train_folder_path, limit=1000)  # Limit to 1000 images for testing
print(f"Loaded {len(train_images)} training images.")

print("Loading validation images...")
validate_images, validate_labels = load_images_from_folder(validate_folder_path, limit=100)  # Limit to 100 images for testing
print(f"Loaded {len(validate_images)} validation images.")

# Generate face encodings for training data
print("Generating face encodings for training data...")
train_encodings = []
train_encoding_labels = []
for img, label in tqdm(zip(train_images, train_labels), total=len(train_images)):
    face_encodings = get_face_encodings(img)
    if face_encodings:
        train_encodings.append(face_encodings[0])
        train_encoding_labels.append(label)
print(f"Generated {len(train_encodings)} face encodings for training data.")

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

# Train the model (using the training set)
def train_model(X_train, y_train):
    return X_train, y_train

# Predict the labels (using the validation set)
def predict(model, X_test):
    X_train, y_train = model
    predictions = []
    for encoding in X_test:
        matches = face_recognition.compare_faces(X_train, encoding)
        if True in matches:
            first_match_index = matches.index(True)
            predictions.append(y_train[first_match_index])
        else:
            predictions.append(None)
    return predictions

# Train the model
model = train_model(train_encodings, train_encoding_labels)

# Predict the labels of the validation set
print("Predicting validation labels...")
y_pred = predict(model, validate_encodings)

# Filter out None values from predictions and corresponding labels
filtered_y_pred = [pred for pred in y_pred if pred is not None]
filtered_validate_labels = [label for pred, label in zip(y_pred, validate_encoding_labels) if pred is not None]

# Calculate the accuracy
accuracy = accuracy_score(filtered_validate_labels, filtered_y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

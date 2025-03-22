import numpy as np
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score

# Load the trained model
model = tf.keras.models.load_model("emotion_vgg16.h5")  # Update model name if different
print("✅ Model Loaded Successfully!")

# Define emotion labels (FER-2013 classes)
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Set path to test dataset folder
test_folder = r"D:\OneDrive\Desktop\nullclass internship\dataset\test"  # Update path

# Read images and labels
true_labels = []  # Store actual labels
predicted_labels = []  # Store predicted labels

for emotion in os.listdir(test_folder):  
    emotion_folder = os.path.join(test_folder, emotion)
    
    if not os.path.isdir(emotion_folder):
        continue  # Skip if not a folder

    # Convert folder name to match label names
    emotion_name = emotion.capitalize()  

    if emotion_name in emotion_labels.values():
        label_index = list(emotion_labels.values()).index(emotion_name)
    else:
        print(f"⚠️ Skipping unknown folder: {emotion}")
        continue

    for img_name in os.listdir(emotion_folder):
        img_path = os.path.join(emotion_folder, img_name)
        
        # Load and preprocess image
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
        img_array = image.img_to_array(img)
        if img_array.shape[-1] == 1:  
            img_array = np.stack((img_array,) * 3, axis=-1)  # Duplicate grayscale channel

        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
        # Predict emotion
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Store results
        true_labels.append(label_index)
        predicted_labels.append(predicted_class)

# Compute accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"✅ Model Accuracy on Test Folder: {accuracy * 100:.2f}%")
import numpy as np
import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError, ImageFile
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import os
import tensorflow_datasets as tfds 
import pytesseract
import cv2
from PIL import Image
import numpy as np
import re
from a import pytesseract_image_to_string
# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to download and preprocess the images from URLs
def download_and_preprocess(image_url):
    try:
        response = requests.get(image_url, timeout=20)  # Increased timeout
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        if img.shape != (128, 128, 3):
            print(f"Image shape mismatch: {img.shape} for URL: {image_url}")
            return None
        return img
    except (UnidentifiedImageError, requests.exceptions.RequestException, OSError) as e:
        print(f"Error processing image {image_url}: {e}")
        return None

# Preload a smaller subset of data into memory for testing
def preload_data(df, label_encoder, num_samples=50):
    images = []
    labels = []
    for _, row in df.sample(n=num_samples).iterrows():
        img = download_and_preprocess(row['image_link'])
        if img is not None:
            images.append(img)
            labels.append(row['entity_value'])
    numeric_labels = label_encoder.transform(labels)
    return np.array(images), np.array(numeric_labels)

# Reload data
train_file_path = '/content/drive/My Drive/student_resource 3/dataset/train.csv'
test_file_path = '/content/drive/My Drive/student_resource 3/dataset/test.csv'

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Print the first few rows of the dataset for verification
print(train_df.head())
print(test_df.head())

# Encode the 'entity_value' column to numeric labels
label_encoder = LabelEncoder()
train_df['entity_value_encoded'] = label_encoder.fit_transform(train_df['entity_value'])

# Preload a smaller subset of data into memory
images, labels = preload_data(train_df, label_encoder)
print("Preloaded data shapes:", images.shape, labels.shape)

# Use tf.data.Dataset for better performance
def create_tf_dataset(images, labels, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

batch_size = 8
train_dataset = create_tf_dataset(images, labels, batch_size)

# Build and compile the model within the TPU strategy scope
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

sample_df = train_df.sample(n=10)  # Smaller subset for testing
images = []
labels = []
for _, row in sample_df.iterrows():
    img = download_and_preprocess(row['image_link'])
    if img is not None:
        images.append(img)
        labels.append(row['entity_value'])

images = np.array(images)
tempimages = images[0]
numeric_labels = label_encoder.transform(labels)
print("Preloaded batch shapes:", images.shape, numeric_labels.shape)

image = cv2.imread('url')
print(pytesseract_image_to_string(tempimages))
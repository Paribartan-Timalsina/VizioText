import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Define parameters
image_size = (256, 256)  # Input size for ResNet50
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # Load ResNet50

# Directory paths
train_dir = "/content/train"
test_dir = "/content/test"
validation_dir = "/content/validation"

def preprocess_image(image_path):
    """Preprocess the image for the CNN."""
    image = load_img(image_path, target_size=image_size)  # Load the image
    image = img_to_array(image)  # Convert the image to an array
    image = np.expand_dims(image, axis=0)  # Add a batch dimension
    image = image / 255.0  # Normalize the image
    return image

def extract_features(image_directory):
    """Extract features from images in the given directory using the CNN."""
    features = {}
    for image_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_file)
        if os.path.isfile(image_path):
            image = preprocess_image(image_path)  # Preprocess the image
            feature = model.predict(image, verbose=0)  # Extract features using CNN
            features[image_file] = feature.flatten()  # Store flattened feature vector
    return features

if __name__ == "__main__":
    # Extract features for each dataset
    train_features = extract_features(train_dir)
    test_features = extract_features(test_dir)
    validation_features = extract_features(validation_dir)

    # Save extracted features
    with open('train_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)

    with open('test_features.pkl', 'wb') as f:
        pickle.dump(test_features, f)

    with open('validation_features.pkl', 'wb') as f:
        pickle.dump(validation_features, f)

    print("Features extracted and saved successfully.")

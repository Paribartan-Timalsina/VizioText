import os
import random
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.preprocessing.image import load_img


image_format=["jpg","png","jpeg"]
root_dir="/"
for image_directory in ["train","test","validation"]:
  image_dir=os.path.join(root_dir,image_directory)
  for image in os.listdir(image_dir):
    if image.split(".")[-1].lower() not in image_format:
      file_path=os.path.join(image_dir,image)
      try:
        os.remove(file_path)
      except Exception as e:
        print(e)

# Define the root directory and paths
train_dir = os.path.join(root_dir, "train")
caption_file = os.path.join(root_dir, "train_caption.txt")

# Read the captions
captions = {}
with open(caption_file, 'r') as file:
    for line in file:
        parts = line.strip().split(',', 1)
        if len(parts) == 2:
            filename, caption = parts
            if filename in captions:
                captions[filename].append(caption)
            else:
                captions[filename] = [caption]



# Define parameters
image_size = (256, 256)
num_images = 5  # Number of images to display

# Function to display images with captions in vertical format
def display_images_vertically(num_images=5):
    image_files = list(captions.keys())[:num_images]  # Select the first few images
    plt.figure(figsize=(30, 20))  # Increase figure size for vertical layout
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(train_dir, image_file)
        image = load_img(image_path, target_size=image_size)
        
        # Display the image
        ax = plt.subplot(num_images, 1, i + 1)
        ax.imshow(image)
        ax.axis('off')
        
        # Display the caption below the image
        caption = random.choice(captions[image_file])
        plt.text(0.5, -0.2, caption, ha='center', va='top', transform=ax.transAxes, fontsize=12, wrap=True)
    
    plt.subplots_adjust(hspace=1)  # Add space between images and captions
    plt.show()

# Display images vertically with captions below each one
display_images_vertically(num_images=5)

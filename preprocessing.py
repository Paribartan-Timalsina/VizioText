import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img

# Define parameters and directories
image_format = ["jpg", "png", "jpeg"]
root_dir = "/"  # Set your root directory accordingly
train_dir = os.path.join(root_dir, "train")
caption_file = os.path.join(root_dir, "train_caption.txt")

# Step 1: Clean up images (remove files that are not images)
def clean_image_directory(root_dir, image_format):
    for image_directory in ["train", "test", "validation"]:
        image_dir = os.path.join(root_dir, image_directory)
        for image in os.listdir(image_dir):
            if image.split(".")[-1].lower() not in image_format:
                file_path = os.path.join(image_dir, image)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")

# Step 2: Read captions from file
def read_captions(caption_file):
    captions = {}
    with open(caption_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                filename, caption = parts
                captions.setdefault(filename, []).append(caption)
    return captions

# Step 3: Display images vertically with captions
def display_images_vertically(captions, train_dir, image_size=(256, 256), num_images=5):
    image_files = list(captions.keys())[:num_images]
    plt.figure(figsize=(30, 20))  # Set the figure size

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(train_dir, image_file)
        image = load_img(image_path, target_size=image_size)
        
        # Display the image
        ax = plt.subplot(num_images, 1, i + 1)
        ax.imshow(image)
        ax.axis('off')

        # Display all captions below the image
        all_captions = "\n".join(captions[image_file])
        plt.text(0.5, -0.2, all_captions, ha='center', va='top', transform=ax.transAxes, fontsize=12, wrap=True)

    plt.subplots_adjust(hspace=1)  # Add space between images and captions
    plt.show()

# Step 4: Count the number of images in each directory and display as a bar graph
def count_images(root_dir):
    numberofimages = {}
    for image_directory in ["train", "test", "validation"]:
        image_dir = os.path.join(root_dir, image_directory)
        numberofimages[image_directory] = len([image for image in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, image))])
    return numberofimages

# Step 5: Plot the number of images per directory as a bar graph
def plot_image_counts(numberofimages):
    plt.figure(figsize=(8, 6))
    plt.bar(numberofimages.keys(), numberofimages.values(), color='skyblue')
    plt.xlabel("Image Directory")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Directory")
    plt.show()

# Main block
if __name__ == "__main__":
    # Clean up non-image files
    clean_image_directory(root_dir, image_format)

    # Read captions from file
    captions = read_captions(caption_file)

    # Display images and captions
    display_images_vertically(captions, train_dir)

    # Count images and display the bar graph
    numberofimages = count_images(root_dir)
    print(numberofimages)
    plot_image_counts(numberofimages)

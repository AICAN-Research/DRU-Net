import cv2
import os
import numpy as np

from src.utils.img_utils import remove_small_fragments, smooth_edges


def process_images_in_directory(directory, size_threshold):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            
            # Remove small fragments
            binary_image = remove_small_fragments(image_path, size_threshold)

            # Smooth the edges
            smoothed_image = smooth_edges(binary_image)

            # Save the processed image
            cv2.imwrite(image_path, smoothed_image)
            print(f"Processed {filename}")

# Define the directory and size threshold
directory = '/Path/To/SegmentationResults/'  # Update with the path to your images
size_threshold = 10  # Update this value based on your requirement

process_images_in_directory(directory, size_threshold)

# post processing

import os

import cv2
import numpy as np


def remove_small_fragments(image_path, size_threshold):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded correctly
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find all contours
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small fragments
    for cnt in contours:
        if cv2.contourArea(cnt) < size_threshold:
            cv2.drawContours(binary_image, [cnt], -1, (0, 0, 0), -1)

    return binary_image


def smooth_edges(binary_image, kernel_size=7, iterations=1):
    # Define the kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological opening (erosion followed by dilation)
    smoothed_image = cv2.medianBlur(binary_image, ksize=11)
    smoothed_image = cv2.morphologyEx(
        smoothed_image, cv2.MORPH_OPEN, kernel, iterations=iterations
    )

    return smoothed_image


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
directory = (
    "/Path/To/SegmentationResults/"  # Update with the path to your images
)
size_threshold = 10  # Update this value based on your requirement

process_images_in_directory(directory, size_threshold)

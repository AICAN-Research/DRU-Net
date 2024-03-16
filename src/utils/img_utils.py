import tensorflow as tf
import numpy as np
import cv2


def adjust_gamma(image, gamma=1.9, rotation_angle=0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(0, 256)], dtype=np.uint8)
    gamma_corrected = cv2.LUT(image, table)
    rotated_image = rotate_image(gamma_corrected, rotation_angle)
    return rotated_image


def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image


def change_hue_chroma_lightness(image, hue_offset=0, chroma_scale=1.0, lightness_scale=1.0):
    # # Convert BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split the HSV channels
    h, s, v = cv2.split(hsv_image)
    # Apply the hue offset
    h = (h + hue_offset) % 180
    # Scale the chroma (saturation) and lightness
    s = np.clip(np.round(s * chroma_scale), 0, 255).astype(np.uint8)
    v = np.clip(np.round(v * lightness_scale), 0, 255).astype(np.uint8)
    # Merge the modified HSV channels
    modified_hsv = cv2.merge((h, s, v))
    # Convert back to BGR color space
    modified_bgr = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

    return modified_bgr


def adjust_color_balance(image, cyan_red, magenta_green, yellow_blue):
    # Split the image into its RGB channels
    blue, green, red = cv2.split(image)
    # Apply color adjustments for each channel
    red = np.clip(red + cyan_red, 0, 255).astype(np.uint8)
    green = np.clip(green + magenta_green, 0, 255).astype(np.uint8)
    blue = np.clip(blue + yellow_blue, 0, 255).astype(np.uint8)
    # Merge the adjusted channels back into a single image
    adjusted_image = cv2.merge([blue, green, red])

    return adjusted_image


def threshold_pixel(pixel_value, threshold_value):
    return pixel_value < threshold_value


def flip_horizontal(image):
    return cv2.flip(image, 1)


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
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    smoothed_image = cv2.morphologyEx(smoothed_image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    return smoothed_image

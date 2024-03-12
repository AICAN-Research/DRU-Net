# data generator 

import tensorflow as tf
import numpy as np
import cv2
import os
import fast

from MLD import multi_lens_distortion


def load_patch(x_start_val_lvl3, y_start_val_lvl3, filename, level, patch_size):
    if not isinstance(filename, str):
        filename = filename.numpy().decode("utf-8")
    # convert coordinates to numpy (necessary due to slicing in tf that is unstable)
    x_start_val_lvl3 = np.asarray(x_start_val_lvl3)
    y_start_val_lvl3 = np.asarray(y_start_val_lvl3)  
    importer = fast.WholeSlideImageImporter.create(filename)
    wsi = importer.runAndGetOutputData()
    patch_access = wsi.getAccess(fast.ACCESS_READ)
    patch_small = patch_access.getPatchAsImage(level, int(x_start_val_lvl3[0]), int(y_start_val_lvl3[0]), patch_size, patch_size, False)
    patch_small = np.asarray(patch_small)

    return np.asarray(patch_small)

## for augmentation
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


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, StartingPositions, Gts, batch_size):
        self.StartingPositions = StartingPositions
        self.Gts = Gts
        self.batch_size = batch_size
    def __len__(self):
        return int(len(self.StartingPositions) / self.batch_size)


def __getitem__(self, index):
    # Existing code to select batch addresses and labels
    batch_addresses = self.StartingPositions[index * self.batch_size:(index + 1) * self.batch_size].copy()
    batch_labels = self.Gts[index * self.batch_size:(index + 1) * self.batch_size].copy()
    batch_images = np.zeros((self.batch_size, 256, 256, 3), dtype="float32")
    labels_batch = np.zeros((self.batch_size, 2), dtype="float32")

    for ndex, address in enumerate(batch_addresses):
        # Load the image
        image = load_patch(address[3], address[4], address[0], 3, 256)
        # Convert numpy image to TensorFlow tensor for certain augmentations
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_saturation(image, 0.7, 1.3)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Convert back to numpy for lens distortion augmentation
        image_np = image.numpy()
        # Apply multi-lens distortion
        image_np = multi_lens_distortion(image_np, num_lenses=4, radius_range=(40, 70), strength_range=(-0.4, 0.4))
        # Normalize the image
        batch_images[ndex] = image_np / 255.0
        labels_batch[ndex, 0] = 1 - batch_labels[ndex]
        labels_batch[ndex, 1] = batch_labels[ndex]

    return batch_images, labels_batch


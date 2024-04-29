import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def fill_holes(binary_img):
    # Copy the image
    im_in = binary_img.copy()

    # Threshold (to ensure binary input)
    th, im_th = cv2.threshold(im_in, 0.45, 1, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image
    im_floodfill = im_th.copy()

    # Mask used for flood filling. Notice the size needs to be 2 pixels larger than the image
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground
    filled_image = im_th | im_floodfill_inv

    return filled_image


def cluster(image_path, weights=[0.6, 0.1, 0.2], fill_the_holes=True):
    # Load image and extract each channel
    image = cv2.imread(image_path)
    Rw1, Rw2, Rw3 = [image[..., i] / 255 for i in range(3)]

    images = [Rw1, Rw2, Rw3]

    scale_percent = 30  # percent of the original size
    width = int(Rw1.shape[1] * scale_percent / 100)
    height = int(Rw1.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resized_images = [
        cv2.resize(img, dim, interpolation=cv2.INTER_AREA) for img in images
    ]

    weighted_images = [
        img * weight for img, weight in zip(resized_images, weights)
    ]

    # Stack all images to create a feature vector for each pixel
    features = np.stack(weighted_images, axis=-1).reshape(-1, 3)

    # Apply KMeans clustering with a consistent initialization and random seed
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
    labels = kmeans.fit_predict(features)

    # Identify the cluster that is closest to white
    white_cluster = np.argmin(
        np.linalg.norm(kmeans.cluster_centers_ - [1, 1, 1], axis=1)
    )

    # If the white cluster is not labeled as '0', swap labels
    if white_cluster != 0:
        labels[labels == 0] = -1  # Temporary change label '0' to '-1'
        labels[labels == white_cluster] = (
            0  # Assign label '0' to the white cluster
        )
        labels[labels == -1] = (
            white_cluster  # Assign previous '0' cluster to 'white_cluster' label
        )

    # Reshape the labels to the image's shape
    labels_2D = labels.reshape(height, width)

    pred = labels_2D.astype(np.uint8)
    pred = cv2.medianBlur(pred, 11)

    if fill_the_holes:
        pred = fill_holes(pred)

    return pred


def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            result = cluster(image_path, fill_the_holes=True)

            # Create the output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Save the result
            output_path = os.path.join(output_folder, "processed_" + filename)
            cv2.imwrite(
                output_path, result * 255
            )  # Scale back up to 0-255 range

            # Optionally display the result
            plt.imshow(result)
            plt.axis("off")
            plt.show()


# Usage
input_folder = "./input_images"
output_folder = "./output_images"
process_images(input_folder, output_folder)

## creating direct clusters of tissues using pretrained models
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.cluster import KMeans
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as img_prep

## Select GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

files = glob.glob("/Path/To/Thumbnails/*.png")
for f in files:
    ff, _ = os.path.splitext(f)
    basename = os.path.basename(ff)
    print(basename)
    # Load the image
    image_path = f
    original_image = Image.open(image_path)
    original_image = np.array(original_image)
    original_image = cv2.resize(
        original_image,
        (int(original_image.shape[1] / 2), int(original_image.shape[0] / 2)),
    )

    # Define the size of the small squares
    square_size = 32

    # Calculate the number of small squares along width and height
    width, height, _ = original_image.shape
    num_squares_x = width // square_size
    num_squares_y = height // square_size

    # Load pre-trained model + higher level layers
    model = VGG19(weights="imagenet", include_top=False)

    # Initialize an array to store feature vectors
    feature_vectors = []

    # Divide the original image into small squares and extract feature vectors
    for i in range(num_squares_x):
        for j in range(num_squares_y):
            # Extract small square from original image
            square = original_image[
                i * square_size : (i + 1) * square_size,
                j * square_size : (j + 1) * square_size,
            ]
            # Preprocess the square
            square = img_prep.img_to_array(square)
            square = np.expand_dims(square, axis=0)
            square = tf.keras.applications.mobilenet.preprocess_input(square)
            # Extract feature vector using MobileNet
            feature_vector = model.predict(square, verbose=-1)
            feature_vectors.append(feature_vector.flatten())

    feature_vectors = np.array(feature_vectors)

    # Apply k-means clustering on the feature vectors
    num_clusters = 7  # Define the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(feature_vectors)
    labels = kmeans.labels_

    # Create an image with clustering result
    clustered_image = np.zeros_like(original_image)
    for i in range(num_squares_x):
        for j in range(num_squares_y):
            label = labels[i * num_squares_y + j]
            color = np.array(plt.cm.rainbow(label / num_clusters)[:3]) * 255
            clustered_image[
                i * square_size : (i + 1) * square_size,
                j * square_size : (j + 1) * square_size,
            ] = color

    cv2.imwrite("/Path/To/Outputs/" + basename + ".png", clustered_image)

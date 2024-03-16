import cv2
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

dst_dir = "Path/To/Gradients/Reults/"
os.makedirs(dst_dir, exist_ok=True)

files = glob.glob('Path/To/Thumbnails/*.png')
# files2 = glob.glob('D:/Bergens/resized2/*.jpg')


def generate_gradients(imgPath):
    # Convert the image to a tensor
    # img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.io.read_file(imgPath)
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)

    # Get the original shape
    original_shape = tf.shape(img)

    resize_factor = 1

    # Calculate the new height and width as tensors based on the resize factor
    new_height = tf.cast(tf.cast(original_shape[0], tf.float32) * resize_factor, tf.int32)
    new_width = tf.cast(tf.cast(original_shape[1], tf.float32) * resize_factor, tf.int32)

    # Resize the image to <resize_factor> of its original size if necessary
    resized_img = tf.image.resize(img, [new_height, new_width])

    # If you want to ensure the output has the same data type as the input
    resized_img = tf.cast(resized_img, tf.float32)

    img = tf.expand_dims(resized_img, axis=0)
    # print(img)
    # Calculate the gradient in the x and y direction
    gradients = tf.image.image_gradients(img)
    gx, gy = gradients[0], gradients[1]
    # Calculate the magnitude and direction of the gradient
    magnitude = tf.sqrt(tf.math.square(gx) + tf.math.square(gy))
    direction = tf.math.atan2(gy, gx)

    # print(magnitude)
    # plt.imshow(magnitude[0,...,0]*255, cmap='gray')
    # plt.quiver(gx[0,...,0], gy[0,...,0])
    # print(gx[0,...,0])
    # plt.imshow(gx[0,...,0], cmap='gray')
    # plt.imshow(gy[0,...,1], cmap='gray')
    # plt.imshow(direction[0,...,2], cmap='gray')

    a = magnitude[0,...,1]/tf.math.reduce_max(magnitude[0,...,1])

    plt.axis('off')
    root, ext = os.path.splitext(f)
    basename = os.path.basename(root)

    b = np.array(a)
    b *= 255.0/b.max() 

    print(type(b))
    # plt.imshow(np.array(b))
    cv2.imwrite(os.path.join(dst_dir, basename + '' + '.png'), np.array(b))


for indx, f in enumerate(files):
    print(indx)
    generate_gradients(f)

import numpy as np
import tensorflow as tf

from ..augmentation.MLD import multi_lens_distortion


def PreProc(img, pred, mask, img_size):
    img = img / 255.0
    pred = pred / 255.0
    mask = mask / 255.0

    img = tf.image.resize(img, img_size)
    pred = tf.image.resize(pred, img_size)
    mask = tf.image.resize(mask, img_size)

    mask = tf.cast(mask > 0.5, tf.float32)

    return img, pred, mask


def Augmentor(img, pred, mask):
    # Apply transformations to both the image and the mask using a fixed seed for each random operation
    seed = np.random.randint(0, 1e6)  # Generate a common seed for this iteration

    # Random flips
    if tf.random.uniform((), seed=seed) > 0.5:
        img = tf.image.flip_left_right(img)
        pred = tf.image.flip_left_right(pred)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform((), seed=seed) > 0.5:
        img = tf.image.flip_up_down(img)
        pred = tf.image.flip_up_down(pred)
        mask = tf.image.flip_up_down(mask)

    if tf.random.uniform((), seed=seed) > 0.5:
        nbr_rot = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=nbr_rot)
        pred = tf.image.rot90(pred, k=nbr_rot)
        mask = tf.image.rot90(mask, k=nbr_rot)

    # Other transformations
    # print(img.shape)  # This should print something like (224, 224, 4) for a 4-channel image.

    augmented_channels = tf.image.random_hue(img, 0.08, seed=seed)
    augmented_channels = tf.image.random_contrast(augmented_channels, 0.7, 1.3, seed=seed)
    augmented_channels = tf.image.random_brightness(augmented_channels, 0.2, seed=seed)
    augmented_channels = tf.image.random_saturation(augmented_channels, 0.7, 1.3, seed=seed)

    distortion_seed = np.random.randint(0, 2**32 - 1)

    # Apply multi_lens_distortion to both the image and the mask
    img = tf.numpy_function(
        multi_lens_distortion,
        [img, 6, (300, 500), (-0.3, 0.5), distortion_seed],
        tf.float32,
    )

    pred = tf.numpy_function(
        multi_lens_distortion,
        [pred, 6, (300, 500), (-0.3, 0.5), distortion_seed],
        tf.float32,
    )

    mask = tf.numpy_function(
        multi_lens_distortion,
        [mask, 6, (300, 500), (-0.3, 0.5), distortion_seed],
        tf.float32,
    )

    return img, pred, mask

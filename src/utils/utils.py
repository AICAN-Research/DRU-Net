import numpy as np


# explicit function to normalize array
def normalize(x):
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


def PreProc(img, pred, mask):
    img = img/255.
    pred = pred /255.
    mask = mask/255.

    img = tf.image.resize(img,IMG_SIZE)
    pred = tf.image.resize(pred,IMG_SIZE)
    mask = tf.image.resize(mask,IMG_SIZE)

    mask = tf.cast(mask > 0.5, tf.float32)

    return img, pred, mask

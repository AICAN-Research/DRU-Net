import os

import cv2
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model
from MLD import multi_lens_distortion
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU


def build_network():
    input_image = Input(shape=(1120, 1120, 3), name="input_image")
    input_pred = Input(shape=(1120, 1120, 1), name="input_pred")

    conv_pred = layers.Conv2D(3, (3, 3), activation="relu", padding="same")(
        input_pred
    )

    combined = layers.Concatenate()([input_image, conv_pred])

    # Block 1
    c1 = layers.Conv2D(4, (3, 3), activation="relu", padding="same")(combined)
    c1 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Block 2
    c2 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Block 3
    c3 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(p2)
    # c3 = layers.Dropout(0.3)(c3)
    c3 = layers.SpatialDropout2D(0.3)(c3)
    c3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Block 4
    c4 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
    c4 = layers.BatchNormalization()(c4)

    # Bottleneck
    bn = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c4)
    bn = layers.BatchNormalization()(bn)

    # Upsampling (decoder) side

    # Block 1
    u1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(bn)
    u1 = layers.Concatenate()([u1, c4])
    u1 = layers.BatchNormalization()(u1)

    # Block 2 of the Upsampling (decoder) side
    u2 = layers.UpSampling2D(size=(2, 2))(u1)
    u2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u2)
    # u2 = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(u2)  # Adjust padding as needed
    u2 = layers.Concatenate()([u2, c3])
    u2 = layers.BatchNormalization()(u2)

    # Block 3
    u3 = layers.UpSampling2D(size=(2, 2))(u2)
    u3 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(u3)
    u3 = layers.Concatenate()([u3, c2])
    u3 = layers.BatchNormalization()(u3)

    # Block 4
    u4 = layers.UpSampling2D(size=(2, 2))(u3)
    u4 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(u4)
    # u4 = layers.Concatenate()([u4, c1_1])
    u4 = layers.BatchNormalization()(u4)

    # Final Layer
    x = layers.Conv2D(2, (3, 3), activation="softmax", padding="same")(u4)

    model = models.Model(inputs=[input_image, input_pred], outputs=x)
    return model


unet_model = build_network()
unet_model.summary()


IMG_SIZE = (1120, 1120)


def PreProc(img, pred, mask):

    img = img / 255.0
    pred = pred / 255.0
    mask = mask / 255.0

    img = tf.image.resize(img, IMG_SIZE)
    pred = tf.image.resize(pred, IMG_SIZE)
    mask = tf.image.resize(mask, IMG_SIZE)

    mask = tf.cast(mask > 0.5, tf.float32)

    return img, pred, mask


def Augmentor(img, pred, mask):
    # Apply transformations to both the image and the mask using a fixed seed for each random operation

    seed = np.random.randint(
        0, 1e6
    )  # Generate a common seed for this iteration

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
        nbr_rot = tf.random.uniform(
            shape=[], minval=1, maxval=4, dtype=tf.int32
        )
        img = tf.image.rot90(img, k=nbr_rot)
        pred = tf.image.rot90(pred, k=nbr_rot)
        mask = tf.image.rot90(mask, k=nbr_rot)

    # Other transformations
    # print(img.shape)  # This should print something like (224, 224, 4) for a 4-channel image.

    augmented_channels = tf.image.random_hue(img, 0.08, seed=seed)
    augmented_channels = tf.image.random_contrast(
        augmented_channels, 0.7, 1.3, seed=seed
    )
    augmented_channels = tf.image.random_brightness(
        augmented_channels, 0.2, seed=seed
    )
    augmented_channels = tf.image.random_saturation(
        augmented_channels, 0.7, 1.3, seed=seed
    )

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


class TrainDataGenerator(Sequence):
    def __init__(
        self, image_dir, pred_dir, mask_dir, batch_size, augmentation=True
    ):
        self.image_dir = image_dir
        self.pred_dir = pred_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(self.image_dir)
        self.batch_size = batch_size
        self.augmentation = augmentation

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def on_epoch_begin(self):
        np.random.shuffle(self.image_filenames)

    def __getitem__(self, index):
        # Get batch of filenames
        batch_files = self.image_filenames[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        batch_imgs = []
        batch_preds = []
        batch_masks = []
        for filename in batch_files:
            # Load 3-channel image
            img = img_to_array(load_img(os.path.join(self.image_dir, filename)))

            # Load the corresponding 1-channel prediction
            pred = img_to_array(
                load_img(
                    os.path.join(self.pred_dir, filename),
                    color_mode="grayscale",
                )
            )

            # Check if prediction has only one channel
            assert (
                pred.shape[2] == 1
            ), f"Prediction {filename} has more than one channel!"

            # Resize prediction to match the image size
            # pred = tf.image.resize(pred, (img.shape[0], img.shape[1]))
            pred = cv2.resize(
                pred,
                (img.shape[0], img.shape[1]),
                interpolation=cv2.INTER_LINEAR,
            )
            pred = np.expand_dims(pred, axis=-1)

            # # Concatenate to form 4-channel input
            # combined_img = np.concatenate([img, pred], axis=-1) # this is for a 4-channel input
            # print(combined_img.shape)
            # Load the corresponding mask
            mask = img_to_array(
                load_img(
                    os.path.join(self.mask_dir, filename),
                    color_mode="grayscale",
                )
            )

            # print(np.unique(img.flatten()))
            # print(np.unique(pred.flatten()))
            # print(np.unique(mask.flatten()))

            # Apply custom preprocessing to both the combined image and the mask
            # combined_img, mask = custom_preprocessing_function(combined_img, mask)
            img, pred, mask = PreProc(img, pred, mask)

            if self.augmentation:
                img, pred, mask = Augmentor(img, pred, mask)

            # print(np.unique(img))
            # print(np.unique(pred))
            # print(np.unique(mask))

            # break

            mask = tf.concat([1 - mask, mask], axis=-1)

            batch_imgs.append(img)
            batch_preds.append(pred)
            batch_masks.append(mask)

        return [np.array(batch_imgs), np.array(batch_preds)], np.array(
            batch_masks
        )


batch_size = 2
train_gen = TrainDataGenerator(
    "./train/images/",
    "./train/pw_predictions/",
    "./train/masks/",
    batch_size=batch_size,
    augmentation=True,
)
val_gen = TrainDataGenerator(
    "./valid/images/",
    "./valid/pw_predictions/",
    "./valid/masks/",
    batch_size=batch_size,
    augmentation=False,
)


def get_dice_loss(nb_classes=1, use_background=False):
    def dice_loss(target, output, epsilon=1e-10):
        smooth = 1.0
        dice = 0
        for i in range(0 if use_background else 1, nb_classes):
            output1 = output[..., i]
            target1 = target[..., i]
            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(
                target1 * target1
            )
            dice += (2.0 * intersection1 + smooth) / (union1 + smooth)
        if use_background:
            dice /= nb_classes
        else:
            dice /= nb_classes - 1
        return tf.clip_by_value(1.0 - dice, 0.0, 1.0 - epsilon)

    return dice_loss


def dsc_thresholded(nb_classes=2, use_background=False):
    def dice(target, output, epsilon=1e-10):
        smooth = 1.0
        dice = 0
        output = tf.cast(output > 0.5, tf.float32)
        for i in range(0 if use_background else 1, nb_classes):
            output1 = output[:, :, :, i]
            target1 = target[:, :, :, i]

            intersection1 = tf.reduce_sum(output1 * target1)
            union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(
                target1 * target1
            )
            dice += (2.0 * intersection1 + smooth) / (union1 + smooth)

        if use_background:
            dice /= nb_classes
        else:
            dice /= nb_classes - 1

        return tf.clip_by_value(dice, 0.0, 1.0 - epsilon)

    return dice


dice_loss_fn = get_dice_loss(nb_classes=2, use_background=False)
dice_thresh_fn = dsc_thresholded()
unet_model.compile(
    optimizer=tf.keras.optimizers.experimental.Adam(1e-4),
    loss=dice_loss_fn,
    metrics=[dice_thresh_fn],
)

early = EarlyStopping(monitor="val_loss", patience=20, verbose=1)

save_best = ModelCheckpoint(
    "./model",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
    period=1,
)

history = unet_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=300,
    callbacks=[early, save_best],
)


best_model = load_model("./model", compile=False)

onnx_model, _ = tf2onnx.convert.from_keras(best_model, opset=13)
onnx.save(onnx_model, "./model.onnx")

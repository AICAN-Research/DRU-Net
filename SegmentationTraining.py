import os
import numpy as np
import cv2
import tensorflow as tf
import tf2onnx
import onnx

from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from MLD import multi_lens_distortion

from src.models.losses import get_dice_loss, dsc_thresholded
from src.models.drunet import build_drunet

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Select GPU

# params
IMG_SIZE = (1120, 1120)
batch_size = 2

# create DRU-Net and compile model
drunet_model = build_network()
drunet_model.summary()

dice_loss_fn = get_dice_loss(nb_classes=2, use_background=False)
dice_thresh_fn = dsc_thresholded()
drunet_model.compile(optimizer=tf.keras.optimizers.experimental.Adam(1e-4), loss=dice_loss_fn, metrics=[dice_thresh_fn])

# setup data generator
train_gen = TrainDataGenerator('./train/images/', './train/pw_predictions/', './train/masks/', batch_size=batch_size, augmentation=True)
val_gen = TrainDataGenerator('./valid/images/', './valid/pw_predictions/', './valid/masks/', batch_size=batch_size, augmentation=False)

early = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

save_best = ModelCheckpoint(
    './model',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)

history = drunet_model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=300,
    callbacks=[early, save_best]
)

best_model = load_model('./model', compile=False)

onnx_model, _ = tf2onnx.convert.from_keras(best_model, opset=13)
onnx.save(onnx_model, "./model.onnx")

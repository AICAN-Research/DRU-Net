import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

from ..augmentation.augmentor import Augmentor
from ..utils.utils import PreProc
from ..augmentation.MLD import multi_lens_distortion


class TrainDataGenerator(Sequence):
    def __init__(self, image_dir, pred_dir, mask_dir, batch_size, augmentation=True):
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
        batch_files = self.image_filenames[index*self.batch_size : (index+1)*self.batch_size]
        
        batch_imgs = []
        batch_preds = []
        batch_masks = []
        for filename in batch_files:
            # Load 3-channel image
            img = img_to_array(load_img(os.path.join(self.image_dir, filename)))
            
            # Load the corresponding 1-channel prediction
            pred = img_to_array(load_img(os.path.join(self.pred_dir, filename), color_mode='grayscale'))
            
            # Check if prediction has only one channel
            assert pred.shape[2] == 1, f"Prediction {filename} has more than one channel!"
            
            # Resize prediction to match the image size
            # pred = tf.image.resize(pred, (img.shape[0], img.shape[1]))
            pred = cv2.resize(pred, (img.shape[0], img.shape[1]), interpolation=cv2.INTER_LINEAR)
            pred = np.expand_dims(pred, axis=-1)

            # # Concatenate to form 4-channel input
            # combined_img = np.concatenate([img, pred], axis=-1) # this is for a 4-channel input
            # Load the corresponding mask
            mask = img_to_array(load_img(os.path.join(self.mask_dir, filename), color_mode='grayscale'))

            # Apply custom preprocessing to both the combined image and the mask
            # combined_img, mask = custom_preprocessing_function(combined_img, mask)
            img, pred, mask = PreProc(img, pred, mask)

            if self.augmentation:
                img, pred, mask = Augmentor(img, pred, mask)

            mask = tf.concat([1 - mask, mask], axis=-1)

            batch_imgs.append(img)
            batch_preds.append(pred)
            batch_masks.append(mask)

        return [np.array(batch_imgs), np.array(batch_preds)], np.array(batch_masks)


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

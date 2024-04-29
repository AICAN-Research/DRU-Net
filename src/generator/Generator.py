import fast
import numpy as np
import tensorflow as tf
from ..augmentation.MLD import multi_lens_distortion


def load_patch(x_start_val_lvl3, y_start_val_lvl3, filename, level, patch_size):
    print("This is the filename: ", filename)
    print("This X start point: ", x_start_val_lvl3)
    print("This Y start point: ", y_start_val_lvl3)

    if not isinstance(filename, str):
        filename = filename.numpy().decode("utf-8")

    x_start_val_lvl3 = np.asarray(x_start_val_lvl3)
    y_start_val_lvl3 = np.asarray(y_start_val_lvl3)

    importer = fast.WholeSlideImageImporter.create(filename)
    wsi = importer.runAndGetOutputData()
    patch_access = wsi.getAccess(fast.ACCESS_READ)
    patch = patch_access.getPatchAsImage(
        level,
        int(x_start_val_lvl3[0]),
        int(y_start_val_lvl3[0]),
        patch_size,
        patch_size,
        False,
    )

    # Direct conversion to numpy array upon creation to avoid redundant calls
    return np.asarray(patch)


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, starting_positions, gts, batch_size, patch_size=256, level=3
    ):
        self.starting_positions = starting_positions
        self.gts = gts
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.level = level

        # Dictionary to hold indices for balancing based on each combination of GT and clustering label
        self.combination_indices = {
            (0, 1): [],
            (0, 2): [],
            (0, 3): [],
            (0, 4): [],
            (1, 1): [],
            (1, 2): [],
            (1, 3): [],
            (1, 4): [],
        }

        # Populate combination_indices with index values
        for i, address in enumerate(self.starting_positions):
            gt = self.gts[i]
            cluster_label = address[
                3
            ]  # Extract cluster_label from starting_positions
            self.combination_indices[(gt, cluster_label)].append(i)

        # Calculate the minimum count across all categories to ensure balance
        self.min_samples = min(
            [len(indices) for indices in self.combination_indices.values()]
        )

    def __len__(self):
        # Each epoch will have a balanced set of samples across all categories
        total_samples = self.min_samples * len(
            self.combination_indices
        )  # Total samples for all categories
        return total_samples // self.batch_size

    def __getitem__(self, idx):
        batch_images = []
        batch_labels = np.zeros((self.batch_size, 2))

        samples_per_category = self.batch_size // len(self.combination_indices)

        # Reset indices for the current batch to ensure balanced distribution
        current_batch_indices = []

        for category, indices in self.combination_indices.items():
            selected_indices = np.random.choice(
                indices, samples_per_category, replace=False
            )
            current_batch_indices.extend(selected_indices)

        np.random.shuffle(
            current_batch_indices
        )  # Shuffle to mix the categories within the batch

        for i, index in enumerate(current_batch_indices):
            position = self.starting_positions[index]
            x_start, y_start, filename, _ = (
                position[1],
                position[2],
                position[0],
                position[3],
            )
            image = load_patch(
                x_start, y_start, filename, self.level, self.patch_size
            )

            # Augmentation
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            image = tf.image.random_hue(image, 0.08)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_saturation(image, 0.7, 1.3)
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            # Convert back to numpy for lens distortion augmentation
            image_np = image.numpy()
            image_np = multi_lens_distortion(
                image_np,
                num_lenses=4,
                radius_range=(40, 70),
                strength_range=(-0.4, 0.4),
            )

            batch_images.append(image_np)
            batch_labels[i, 0] = 1 - self.gts[index]
            batch_labels[i, 1] = self.gts[index]

        return np.array(batch_images), batch_labels

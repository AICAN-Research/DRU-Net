import os
import numpy as np
import cv2
import tensorflow as tf
import tf2onnx
import onnx

from keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from PIL import Image
from PIL import ImageEnhance
from MLD import multi_lens_distortion

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select GPU with index 0


filenames_Tumor = next(os.walk('./Path/To/Tumor/'), (None, None, []))[2]  # [] if no file
filenames_Normal = next(os.walk('./Path/To/Normal/'), (None, None, []))[2]  # [] if no file

data = {
    "Tumor":['./Path/To/Tumor/'+i for i in filenames_Tumor],
    "Normal":['./Path/To/Normal/'+j for j in filenames_Normal]

}


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image)[:,:,0:3]
    # print(image)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_crop(image, (224, 224, 3))
    image = tf.cast(image, tf.float32)
    image = image/255
    # image = tf.numpy_function(
    # multi_lens_distortion, 
    # [image, 4, (80, 110), (-0.5, 0.5)], 
    # tf.uint8
    # )
    return image

data_images = {
    "Tumor": data["Tumor"],
    "Normal": data["Normal"]
}

def load_images(paths):
    return np.array([load_image(path) for path in paths])

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

def embedding_model():
    prev_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

    z = tf.keras.layers.Flatten()(prev_model.output)
    z = tf.keras.layers.Dense(32, activation="relu")(z)
    z = tf.keras.layers.Dense(2, activation="softmax")(z)
    return tf.keras.Model(prev_model.input, outputs=z)


embedding_net = embedding_model()
for layer in embedding_net.layers[:-12]:
    layer.trainable = False
embedding_net.compile(optimizer=optimizers.Adam(0.1), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


def compute_prototype(embeddings, labels):
    class_embeddings = tf.math.reduce_mean(embeddings[labels], axis=0)
    return class_embeddings


n_shots = 2
n_query = 3

num_epochs = 100

best_loss = float('inf')  # Initialize best loss to infinity
best_model_path = 'best_model.h5'  # Define path to save the best model

for epoch in range(num_epochs): 
    epoch_loss_avg = tf.keras.metrics.Mean()
    # Randomly sample support set and query set for both classes
    support_idx_tumor = np.random.choice(len(data_images["Tumor"]), n_shots, replace=False)
    query_idx_tumor = np.random.choice(len(data_images["Tumor"]), n_query, replace=False)
    
    support_idx_normal = np.random.choice(len(data_images["Normal"]), n_shots, replace=False)
    query_idx_normal = np.random.choice(len(data_images["Normal"]), n_query, replace=False)

    # Load images using indices and paths
    support_tumor = load_images([data_images["Tumor"][i] for i in support_idx_tumor])
    query_tumor = load_images([data_images["Tumor"][i] for i in query_idx_tumor])
    
    support_normal = load_images([data_images["Normal"][i] for i in support_idx_normal])
    query_normal = load_images([data_images["Normal"][i] for i in query_idx_normal])
    

    support_set = tf.concat([support_normal, support_tumor ], axis=0)
    query_set = tf.concat([query_normal, query_tumor], axis=0)
    

    support_labels = [0] * n_shots + [1] * n_shots
    query_labels = [0] * n_query + [1] * n_query

    # Ensure labels are one-hot encoded
    
    query_labels_one_hot = tf.one_hot(query_labels, depth=2)

    support_embeddings = embedding_net(support_set)
    query_embeddings = embedding_net(query_set)
    

    tumor_prototype = compute_prototype(support_embeddings, tf.equal(support_labels, 1))
    normal_prototype = compute_prototype(support_embeddings, tf.equal(support_labels, 0))
    # print(tumor_prototype.shape)

    prototypes = tf.stack([tumor_prototype, normal_prototype])

    # Compute Euclidean distance from each query embedding to the prototypes
    distances = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=-1)


    # Optimize
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    # optimizer.minimize(loss, embedding_net.trainable_variables)

    # Compute the loss and optimize
    with tf.GradientTape() as tape:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=-distances, labels=query_labels_one_hot))

        epoch_loss_avg.update_state(loss)
        # All model-related calculations here
        support_embeddings = embedding_net(support_set)
        query_embeddings = embedding_net(query_set)
        
        tumor_prototype = compute_prototype(support_embeddings, tf.equal(support_labels, 1))
        normal_prototype = compute_prototype(support_embeddings, tf.equal(support_labels, 0))
        prototypes = tf.stack([tumor_prototype, normal_prototype])
        
        distances = tf.norm(tf.expand_dims(query_embeddings, 1) - prototypes, axis=-1)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=-distances, labels=query_labels_one_hot))

    print(f"Epoch {epoch+1}: Loss: {epoch_loss_avg.result()}")
    gradients = tape.gradient(loss, embedding_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, embedding_net.trainable_variables))

    # Check if the current epoch's loss is lower than the best recorded loss
    current_loss = epoch_loss_avg.result().numpy()
    if current_loss < best_loss:
        best_loss = current_loss
        # Save the model if it has the best loss so far
        embedding_net.save(best_model_path)
        print(f"Model saved at Epoch {epoch+1} with loss: {best_loss}")


best_model = tf.keras.models.load_model(best_model_path)

onnx_model, _ = tf2onnx.convert.from_keras(best_model, opset=13)
onnx.save(onnx_model, "./FewShotModel.onnx")
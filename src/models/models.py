import tensorflow as tf
from tensorflow.keras import layers, models, Input


def build_drunet():
    input_image = Input(shape=(1120, 1120, 3), name='input_image')
    input_pred = Input(shape=(1120, 1120, 1), name='input_pred')

    conv_pred = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(input_pred)

    combined = layers.Concatenate()([input_image, conv_pred])
    
    # Block 1
    c1 = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(combined)
    c1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Block 2
    c2 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(p2)
    # c3 = layers.Dropout(0.3)(c3)
    c3 = layers.SpatialDropout2D(0.3)(c3)
    c3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    # Block 4
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    
    # Bottleneck
    bn = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    bn = layers.BatchNormalization()(bn)
    
    # Upsampling (decoder) side

    # Block 1
    u1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(bn)
    u1 = layers.Concatenate()([u1, c4])
    u1 = layers.BatchNormalization()(u1)

    # Block 2 of the Upsampling (decoder) side
    u2 = layers.UpSampling2D(size=(2, 2))(u1)
    u2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u2)
    # u2 = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(u2)  # Adjust padding as needed
    u2 = layers.Concatenate()([u2, c3])
    u2 = layers.BatchNormalization()(u2)

    # Block 3
    u3 = layers.UpSampling2D(size=(2, 2))(u2)
    u3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u3)
    u3 = layers.Concatenate()([u3, c2])
    u3 = layers.BatchNormalization()(u3)

    # Block 4
    u4 = layers.UpSampling2D(size=(2, 2))(u3)
    u4 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(u4)
    # u4 = layers.Concatenate()([u4, c1_1])
    u4 = layers.BatchNormalization()(u4)
    
    # Final Layer
    x = layers.Conv2D(2, (3, 3), activation='softmax', padding='same')(u4)

    return models.Model(inputs=[input_image,input_pred], outputs=x)


def embedding_model(img_shape=(224, 224, 3)):
    prev_model = tf.keras.applications.DenseNet121(input_shape=img_shape, include_top=False, weights='imagenet')

    z = tf.keras.layers.Flatten()(prev_model.output)
    z = tf.keras.layers.Dense(32, activation="relu")(z)
    z = tf.keras.layers.Dense(2, activation="softmax")(z)
    return tf.keras.Model(prev_model.input, outputs=z)

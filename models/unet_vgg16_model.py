# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2

def unet_vgg16_model(input_shape=(256, 256, 3), num_classes=1):

    """
    U-Net model with a VGG16 encoder for image segmentation.

    Notes:

    (1) L2 Regularization penalizes large weights
    and SpatialDropout2D randomly drops feature maps.
    Both methods help prevent overfitting.

    (2) kernel_initializer = "he_normal" is recommended
    for layers with relu activation.
    """

    inputs = layers.Input(shape=input_shape)

    # Load the VGG16 model with pre-trained imagenet weights
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg16.trainable = False # VGG16 layers are frozen, switch to True to update those weights

    # Encoder: Extract features from VGG16
    conv1 = vgg16.get_layer("block1_conv2").output  # 64 channels
    conv2 = vgg16.get_layer("block2_conv2").output  # 128 channels
    conv3 = vgg16.get_layer("block3_conv3").output  # 256 channels
    conv4 = vgg16.get_layer("block4_conv3").output  # 512 channels
    conv5 = vgg16.get_layer("block5_conv3").output  # 512 channels

    # Decoder: Expanding path
    up6 = layers.Conv2DTranspose(filters = 512, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up6)
    conv6 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv6)
    
    up7 = layers.Conv2DTranspose(filters = 256, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up7)
    conv7 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv7)
    
    up8 = layers.Conv2DTranspose(filters = 128, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up8)
    conv8 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv8)
    

    up9 = layers.Conv2DTranspose(filters = 64, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up9)
    conv9 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv9)
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(filters = 1, kernel_size = (3, 3), padding="same", activation = "sigmoid")(conv9)
    else:
        outputs = layers.Conv2D(filters = 1, kernel_size = (3, 3), padding="same", activation = "softmax")(conv9)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

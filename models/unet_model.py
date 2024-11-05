# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

def unet_model(input_shape=(256, 256, 1), num_classes=1):

    """
   U-Net model for image segmentation.
    
    Notes: 
        
    (1) L2 Regularization penalizes large weights
    and SpatialDropout2D randomly drops feature maps.
    Both methods help prevent overfitting.
    
    (2) kernel_initializer = "he_normal" is recommended 
    for layers with relu activation.

    (3) For more information regarding the U-Net architecture,
    the interested reader is referred to the work of Olaf Ronneberger et al.,
    "U-Net: Convolutional Networks for Biomedical Image Segmentation".
    """

    inputs = layers.Input(shape=input_shape)

    # Encoder: Contracting path
    conv1 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(inputs)
    conv1 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv1)
    conv1 = layers.SpatialDropout2D(0.1)(conv1)  # 10% spatial dropout
    pool1 = layers.MaxPooling2D(pool_size = (2, 2))(conv1)
    
    conv2 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(pool1)
    conv2 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv2)
    conv2 = layers.SpatialDropout2D(0.2)(conv2)  # 20% spatial dropout
    pool2 = layers.MaxPooling2D(pool_size = (2, 2))(conv2)
    
    conv3 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(pool2)
    conv3 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv3)
    conv3 = layers.SpatialDropout2D(0.4)(conv3)  # 20% spatial dropout
    pool3 = layers.MaxPooling2D(pool_size = (2, 2))(conv3)
    
    conv4 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(pool3)
    conv4 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv4)
    conv4 = layers.SpatialDropout2D(0.4)(conv4)  # 40% spatial dropout
    pool4 = layers.MaxPooling2D(pool_size = (2, 2))(conv4)
    
    conv5 = layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(pool4)
    conv5 = layers.Conv2D(filters = 1024, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv5)
    conv5 = layers.SpatialDropout2D(0.5)(conv5)  # 20% spatial dropout

    # Decoder: Expanding path
    up6 = layers.Conv2DTranspose(filters = 512, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv5)
    up6 = layers.concatenate([up6, conv4])
    conv6 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up6)
    conv6 = layers.Conv2D(filters = 512, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv6)
    conv6 = layers.SpatialDropout2D(0.4)(conv6)  # 40% spatial dropout
    
    up7 = layers.Conv2DTranspose(filters = 256, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv6)
    up7 = layers.concatenate([up7, conv3])
    conv7 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up7)
    conv7 = layers.Conv2D(filters = 256, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv7)
    conv7 = layers.SpatialDropout2D(0.3)(conv7)  # 30% spatial dropout
    
    up8 = layers.Conv2DTranspose(filters = 1282, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv7)
    up8 = layers.concatenate([up8, conv2])
    conv8 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up8)
    conv8 = layers.Conv2D(filters = 128, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv8)
    conv8 = layers.SpatialDropout2D(0.2)(conv8)  # 20% spatial dropout
    
    up9 = layers.Conv2DTranspose(filters = 64, kernel_size = (2, 2), strides = (2, 2), padding="same")(conv8)
    up9 = layers.concatenate([up9, conv1])
    conv9 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(up9)
    conv9 = layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), padding = "same", activation = "relu", kernel_initializer = "he_normal", kernel_regularizer=l2(0.001))(conv9)
    conv9 = layers.SpatialDropout2D(0.1)(conv9)  # 10% spatial dropout
    
    # Output layer
    if num_classes == 1:
        outputs = layers.Conv2D(filters = 1, kernel_size = (3, 3), padding="same", activation = "sigmoid")(conv9) # for binary segmentation
    else:
        outputs = layers.Conv2D(filters = 1, kernel_size = (3, 3), padding="same", activation = "softmax")(conv9) # for multi-class segmentation (under-development)
        

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

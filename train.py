# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from models.unet_model import unet_model
from models.unet_vgg16_model import unet_vgg16_model
from utils.data_loader import load_images_and_masks, create_datasets
from config.train_config import train_config

def main():

    # Configuration for the training
    config = train_config

    # User-defined inputs
    image_folder = config['image_folder']
    mask_folder = config['mask_folder']
    batch_size = config['batch_size']
    epochs = config['epochs']
    patience = config['patience']
    learning_rate = config['learning_rate']
    model_type = config['model']
    num_classes = config['num_classes']
    augment = config['augment']
    augmentation_size = config['augmentation_size']
    monitoring = config['monitoring']
    log_dir = config['log_dir']

    # Ensures log_dir exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Load images and masks using the default image size from the dataset
    images, masks = load_images_and_masks(image_folder, mask_folder, model_type=model_type)

    # Create datasets
    train_dataset, val_dataset, test_dataset, len_train_dataset = create_datasets(images, masks, batch_size=batch_size, validation_split=0.1, test_split=0.1, augment=augment, augmentation_size=augmentation_size)

    # Initialize the model based on the selected type
    if model_type == 'unet':
        model = unet_model(input_shape=(images.shape[1], images.shape[2], images.shape[3]), num_classes=num_classes)
    elif model_type == 'unet_vgg16':
        model = unet_vgg16_model(input_shape=(images.shape[1], images.shape[2], images.shape[3]), num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compile the model
    if num_classes == 1: # using num_classes == 1 --> assuming binary segmentation
        loss = 'binary_crossentropy'
    else:
        raise ValueError(f"Multi-class segmentation not currently supported. Choose: num_classes = 1")

    """
    else: # using num_classes > 1 --> Multi-class segmentation not sopported yet
        loss = 'categorical_crossentropy'
    """

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=['accuracy'])


    # Callbacks for early stopping and model checkpoints
    callbacks = [
        EarlyStopping(patience=patience, verbose=1),
        ModelCheckpoint(f'models/saved_models/best_{model_type}_model.keras', verbose=1, save_best_only=True, save_weights_only=False),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    ]

    # Add the monitoring callback only if monitoring is True
    if monitoring:
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True))

    # Train the model
    steps_per_epoch = len_train_dataset // batch_size

    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_dataset,
              callbacks=callbacks)

if __name__ == '__main__':
    if not os.path.exists('models/saved_models'):
        os.makedirs('models/saved_models')
    main()

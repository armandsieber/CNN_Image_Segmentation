# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_images_and_masks(image_folder, mask_folder, image_size=None, model_type='unet'):
    """
    Load/preprocess images and masks.

    Arguments:
        image_folder (str): Directory containing the images.
        mask_folder (str): Directory containing the masks.
        image_size (tuple): Size to resize the images and masks (default is None, will use original size).
        model_type (str): Type of model ('unet' or 'unet_vgg16').

    Returns:
        Arrays of preprocessed images and masks.
    """

    # Get list of image and mask filenames
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # If image_size is not defined, compute image_size from first image
    if image_size is None:
        first_image_path = os.path.join(image_folder, image_files[0])
        first_image = Image.open(first_image_path).convert('L')
        image_size = first_image.size

    images = []
    masks = []

    # Load and preprocess each image and mask
    for img_filename, mask_filename in zip(image_files, mask_files):
        img_path = os.path.join(image_folder, img_filename)
        mask_path = os.path.join(mask_folder, mask_filename)
        
        # Open image and mask
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Convert to grayscale
        image = image.convert('L')
        mask = mask.convert('L')

        # Resize (so that all images have the same size)
        image = image.resize(image_size)
        mask = mask.resize(image_size)

        # Re-shape images and masks for compatibility with the models --> to be updated for multi-class segmentation
        if model_type == 'unet_vgg16':
            image = np.stack([image] * 3, axis=-1) # image conversion to (height, width, 3) for compatibility with vgg16 backbone
        elif model_type == 'unet':
            image = np.expand_dims(image, axis=-1) # image conversion to (height, width, 1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        mask = np.expand_dims(mask, axis=-1) # masks conversion to (height, width, 1)

        # Append images and masks
        images.append(np.array(image))
        masks.append(np.array(mask))

    # Normalize the images and masks assuming uint8 images --> needs to be changed if working with different bit-depths
    images = np.array(images) / 255.0
    masks = np.array(masks) / 255.0

    return images, masks

def augment_data(images, masks, augmentation_size):
    """
    Apply data augmentation to images and masks.

    Arguments:
        images (np.array): Array of images.
        masks (np.array): Array of masks.
        augmentation_size (int): Number of augmented versions per image.

    Returns:
        Arrays of original and augmented (augmentation_size times) images and masks.
    """
    datagen = create_datagen()

    # Initialize lists with original images and masks
    augmented_images = list(images)  # Start with the original images
    augmented_masks = list(masks)    # Start with the original masks

    for _ in range(augmentation_size):
        seed = np.random.randint(0, 10000)

        # Ensure the same transformations are applied to images and masks by using the same seed
        image_gen = datagen.flow(images, batch_size=len(images), seed=seed)
        mask_gen = datagen.flow(masks, batch_size=len(masks), seed=seed)

        augmented_images.extend(next(image_gen))
        augmented_masks.extend(next(mask_gen))

    # Convert lists to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)

    return augmented_images, augmented_masks

def create_datagen():
    """
    Create an ImageDataGenerator with specified transformations.
    
    Returns:
        Configured ImageDataGenerator.
    """
    return ImageDataGenerator(
        rotation_range=20,          # Randomly rotates the image within the range of -20 to 20 degrees
        width_shift_range=0.1,      # Randomly shifts the image horizontally by up to 10% of the width
        height_shift_range=0.1,     # Randomly shifts the image vertically by up to 10% of the height
        shear_range=0.1,            # Applies a shear transformation with a magnitude of 0.1 radians
        zoom_range=0.1,             # Randomly zooms in or out by up to 10%
        horizontal_flip=True,       # Randomly flips the image horizontally with a 50% chance
        vertical_flip=True,         # Randomly flips the image vertically with a 50% chance
        fill_mode='reflect'         # Fills empty pixels created by transformations with mirroring
)

def create_datasets(images, masks, batch_size=16, validation_split=0.1, test_split=0.1, augment=False, augmentation_size=5):
    """
    Create training, validation, and test datasets from images and masks.

    Arguments:
        images (np.array): Array of images.
        masks (np.array): Array of masks.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        augment (bool): Whether to use data augmentation
        augmentation_size (int): Number of augmented versions per image.

    Returns:
        Training, validation, and test datasets and train datasets length.
    """
    data_size = len(images)
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    
    test_size = int(data_size * test_split)
    val_size = int(data_size * validation_split)
    train_size = data_size - val_size - test_size

    # Split and create datasets for training, validation, and testing
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_images, train_masks = images[train_indices], masks[train_indices]
    val_images, val_masks = images[val_indices], masks[val_indices]
    test_images, test_masks = images[test_indices], masks[test_indices]

    # Save test images and masks indices (to be reused in evaluate.py)
    np.save('test_indices.npy', test_indices)

    # Apply data augmentation if true
    if augment:
        train_images, train_masks = augment_data(train_images, train_masks, augmentation_size)
        val_images, val_masks = augment_data(val_images, val_masks, augmentation_size)

    # Amount of images in th train dataset
    len_train_dataset = len(train_images)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, len_train_dataset

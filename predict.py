# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.unet_model import unet_model
from models.unet_vgg16_model import unet_vgg16_model
from config.predict_config import predict_config


def predict(model, image):
    """
    Predict the mask for an input image.

    Arguments:
        model (tf.keras.Model): Trained model.
        image (np.array): Input image.

    Returns:
        Predicted mask.
    """

    # Pre-process image
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255

    # Predict mask
    output = model.predict(image)

    # Post-process output
    output = output.squeeze()
    output = (output > 0.5).astype(np.uint8)

    return output

def save_prediction(output, output_path):
    """
    Save the predicted mask to a file.

    Arguments:
        output (np.array): Predicted mask.
        output_path (str): Path to save the mask.
    """
    output_image = Image.fromarray(output * 255)
    output_image.save(output_path)

def plot_results(original_image, prediction):
    """
    Plot the original image and the predicted mask.

    Arguments:
        original_image (np.array): Original image.
        prediction (np.array): Predicted mask.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')

    ax2.imshow(prediction, cmap='gray')
    ax2.set_title('Predicted Mask')

    plt.show()

def main():

    # Configuration for the prediction
    config = predict_config

    # User-defined inputs
    image_path = config['image_path']
    image_size = config['image_size']
    model_type = config['model_type']
    model_path = config['model_path']
    output_path = config['output_path']

    # Load image and the trained model
    image = Image.open(image_path).convert('L')
    image = image.resize(image_size)

    # Load the trained model
    if model_type == 'unet':
        image = np.expand_dims(image, axis=-1)
        input_shape=(image_size[0], image_size[1], 1)
        model = unet_model(input_shape=input_shape, num_classes=1)
    elif model_type == 'unet_vgg16':
        image = np.stack([image] * 3, axis=-1)
        input_shape=(image_size[0], image_size[1], 3)
        model = unet_vgg16_model(input_shape=input_shape, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_weights(model_path)

    output = predict(model, image)
    #save_prediction(output, output_path)
    #print(f"Prediction saved to {output_path}")
    plot_results(image, output)

if __name__ == '__main__':
    main()

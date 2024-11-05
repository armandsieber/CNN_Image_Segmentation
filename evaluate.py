# Copyright (c) 2024 Armand Sieber
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.data_loader import load_images_and_masks
from config.evaluate_config import evaluate_config
from models.unet_model import unet_model
from models.unet_vgg16_model import unet_vgg16_model

def calculate_metrics(true, pred):
    """
    Calculate precision, recall, F1 score, Dice coefficient, and accuracy.

    Arguments
        true (np.array): Ground truth masks.
        pred (np.array): Predicted masks.

    Returns:
        Precision, recall, F1 score and accuracy.
    """
    precision = precision_score(true, pred, average='binary') #  ratio tp / (tp + fp)
    recall = recall_score(true, pred, average='binary') # ratio tp / (tp + fn)
    f1 = f1_score(true, pred, average='binary') # harmonic mean of the precision and recall
    accuracy = accuracy_score(true, pred)

    return precision, recall, f1, accuracy

def evaluate(model, dataset):
    """
    Evaluate the model on a test dataset.

    Arguments:
        model (tf.keras.Model): Trained model.
        dataset (tf.data.Dataset): Test dataset to evaluate on.

    Returns:
        Average precision, recall, F1 score and accuracy.
    """
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    for images, masks in dataset:
        outputs = model.predict(images)
        preds = (outputs > 0.5).astype(np.uint8)

        for i in range(len(images)):
            true_mask = np.array(masks[i]).astype(np.uint8).flatten()
            pred_mask = preds[i].flatten()
            precision, recall, f1, accuracy = calculate_metrics(true_mask, pred_mask)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    return avg_precision, avg_recall, avg_f1, avg_accuracy

def main():

    # Configuration for the evaluation
    config = evaluate_config

    # User-defined inputs
    image_folder = config['image_folder']
    mask_folder = config['mask_folder']
    model_type = config['model_type']
    model_path = config['model_path']
    test_indices_file = config['test_indices_file']
    batch_size = config['batch_size']

    # Load test images and masks
    images, masks = load_images_and_masks(image_folder, mask_folder, model_type=model_type)
    
    # Create dataset
    test_indices = np.load(test_indices_file) # load indices of the test Dataset (the file is generated during training)
    test_images, test_masks = images[test_indices], masks[test_indices]
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Load the trained model
    input_shape=(images.shape[1], images.shape[2], images.shape[3])

    if model_type == 'unet':
        model = unet_model(input_shape=input_shape, num_classes=1)
    elif model_type == 'unet_vgg16':
        model = unet_vgg16_model(input_shape=input_shape, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_weights(model_path)

    # Evaluate the model (custom evaluation, i.e. without using TensorFlow model.evaluate())
    avg_precision, avg_recall, avg_f1, avg_accuracy = evaluate(model, test_dataset)

    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Avg Recall: {avg_recall:.4f}")
    print(f"Avg F1 Score: {avg_f1:.4f}")
    print(f"Avg Accuracy: {avg_accuracy:.4f}")

if __name__ == '__main__':
    main()

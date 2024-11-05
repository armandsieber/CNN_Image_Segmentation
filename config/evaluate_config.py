evaluate_config = {
    'image_folder': 'data/membrane/train_images/',                   # Path to the test images (test images selected from the complete dataset using test_indices_file)
    'mask_folder': 'data/membrane/train_labels/',                    # Path to the test masks (test images selected from the complete dataset using test_indices_file)
    'test_indices_file': 'test_indices.npy',                         # Path to the file with indices of test images (created during training)
    'model_path': 'models/saved_models/best_unet_vgg16_model.keras', # Path to the trained model
    'batch_size': 1,                                                 # Batch size for evaluation
    'model_type': 'unet_vgg16'                                       # Model type: 'unet' or 'unet_vgg16'
}


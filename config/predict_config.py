predict_config = {
    'image_path': 'data/membrane/test_images/test-images0000.tif',      # Path to the input image
    'image_size': (512, 512),                                           # Image size (should be same size as training DataSet)
    'model_path': 'models/saved_models/best_unet_vgg16_model.keras',    # Path to the trained model
    'output_path': 'output_mask.png',                                   # Path to save the predicted mask
    'model_type': 'unet_vgg16'                                          # Model type: 'unet' or 'unet_vgg16'
}

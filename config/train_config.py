train_config = {
    'image_folder': 'data/membrane/train_images',   # Path to training images
    'mask_folder': 'data/membrane/train_labels',    # Path to training masks
    'batch_size': 1,                                # Batch size for training
    'learning_rate': 1e-4,                          # Learning rate for the optimizer
    'epochs': 5,                                    # Number of epochs to train
    'patience': 10,                                 # Patience for early stopping
    'model': 'unet_vgg16',                          # Model type: 'unet' or 'unet_vgg16'
    'num_classes': 1,                               # 1 for binary segmentation. Multi-class segmentation under development
    'augment': False,                               # Whether to apply data augmentation
    'augmentation_size': 8,                         # Number of augmented versions per image.
    'monitoring': True,                             # Enable TensorBoard monitoring if True
    'log_dir': f'./logs/fit/unet_vgg16'             # Path to log files directory (if using TensorBoard monitoring)
}

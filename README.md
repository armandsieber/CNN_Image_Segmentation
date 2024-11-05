# U-Net model for Image Segmentation

## Overview
This project implements a U-Net model in Python using TensorFlow/Keras for 8bit images binary segmentation tasks. A Multi-class segmentation version is under development. U-Net is a type of convolutional neural network (CNN) that can yield precise and effective image segmentation with a small training dataset.

## Project Structure
The project is organized as follows:

```
project/
|-- models/
|   |-- unet_model.py         # U-Net model 
|   |-- unet_vgg16_model.py   # U-Net model with VGG16 backbone
|
|-- utils/
|   |-- data_loader.py        # Functions for loading and augmenting data
|
|-- config/
|   |-- train_config.py       # Configuration for training parameters
|   |-- evaluate_config.py    # Configuration for model accuracy evaluation parameters
|   |-- predict_config.py     # Configuration for prediction parameters (using a trained model)
|
|-- train.py                  # Script to train the model
|-- evaluate.py               # Scipt to evaluate the model accuracy
|-- predict.py                # Script to use the model to segemnt images
|-- requirements.txt          # Project requirements
|-- README.md                 # Project documentation 
|-- License.md 	              # Project license
```

## U-Net Model Architecture
The U-Net architecture consists of an encoder-decoder structure:
- **Encoder**: Extracts features using a series of convolutional and max-pooling layers.
- **Bottleneck**: Connects the encoder and decoder.
- **Decoder**: Uses transposed convolutions and skip connections from the encoder to reconstruct the segmented output.

The architecture implemented in `unet_model.py` can be customized by modifying the number of filters, layers, and activation functions. The project also includes `unet_vgg16_model.py` which performs segmentation using an U-Net
architecture with a pre-trained VGG16 backbone.

For more information regarding the model, the interested reader is referred to the work of Olaf Ronneberger et al. who originally proposed the U-Net architecture, *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*.



## Requirements
The project was developed using Python 3.12.4. The required libraries, along with the versions tested, are listed in the `requirements.txt` file.

## Training Configuration and Data Preparation

### Input Parameters
The training parameters are defined in `train_config.py`:
```python

train_config = {
    'image_folder': 'data/membrane/train_images',   # Path to training images
    'mask_folder': 'data/membrane/train_labels',    # Path to training masks
    'batch_size': 1,                                # Batch size for training
    'learning_rate': 1e-4,                          # Learning rate for the optimizer
    'epochs': 5,                                    # Number of epochs to train
    'patience': 10,                                 # Patience for early stopping
    'model': 'unet',                                # Model type: 'unet' or 'unet_vgg16'
    'num_classes': 1,                               # 1 for binary segmentation. Multi-class segmentation under development
    'augment': True,                                # Whether to apply data augmentation
    'augmentation_size': 8,                         # Number of augmented versions per image.
    'monitoring': True,                             # Enable TensorBoard monitoring if True
    'log_dir': f'./logs/fit/unet'                   # Path to log files directory (if using TensorBoard monitoring)
}
```
### Input Data
The model expects images and masks as input for training:
- **Images**: Should be in a directory specified by `image_folder` in `train_config.py`.
- **Masks**: Corresponding masks should be in a directory specified by `mask_folder`.

Ensure that images and masks have the same dimensions and that both are 8bit grayscale images. Support for images with different bit-depths can be 
achieved with slight modifications to the source code.

### Data Loader
The `data_loader.py` script includes functions to:
- Load images and masks.
- Apply data augmentation for better generalization.
- Create TensorFlow datasets for training and validation.

## Training the Model
To train the U-Net model from the command line, run the `train.py` script:
```bash
python train.py
```
The script will:
- Load and preprocess the image and masks datasets.
- Create training, validation and test datasets.
- Store references to the test datasets in `test_indices.npy` for later use in `evaluate.py`.
- Compile and train the U-Net model using the specified configuration.
- Save the best model to `models/saved_models/`.

## Measurements and Visualizations of Workflow

To track the machine learning workflow, `monitoring` can be enabled in `train_config.py`. The monitoring is achieved
using TensorFlow's visualization and measurement tool TensorBoard.

If you have installed TensorBoard via `pip`, you may launch it from the command line:

```bash
tensorboard --logdir=<path to your logs directory>
```

The command will print an URL that can be opened in your browser (a slightly different procedure must be followed to run TensorBoard on remote server). 
## Model Evaluation and Usage
Once the model is trained, you can assess its performances on a test dataset with the `evaluate.py` script. Its computes 
different metrics such as the precision, recall, f1 scores and accuracy to estimate how well the model performs on unseen data.
In this script, the `test_indices.npy` file contains the indices of the test images that were selected and set aside for testing during the training phase.

Finally, the script `predict.py` may be used to generate segmentation predictions on new input images.


## License
This project is open-source and available under the MIT License.



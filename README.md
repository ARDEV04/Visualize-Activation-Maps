# CNN Activation Visualization

## Overview
This project provides an interactive visualization of Convolutional Neural Network (CNN) activations. It helps understand how different layers in a CNN interpret input images by displaying feature maps and activation patterns at various depths.

## Features
- Load pre-trained CNN models (e.g., VGG16, ResNet, or custom models)
- Visualize activations for any convolutional layer
- Interactive interface for selecting layers and inputs
- Support for different types of input images
- Heatmaps and feature maps for better interpretability

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 
- TensorFlow/Keras or PyTorch
- Matplotlib
- OpenCV
- NumPy

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/ARDEV04/Visualize-Activation-Maps
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
📦 Emotion Detection Heat Map
 ┣ 📂 dataset
 ┃ ┣ 📂 test
 ┃ ┣ 📂 train
 ┣ 📄 .gitattributes
 ┣ 📄 1stprj.py
 ┣ 📄 emotion_vgg16.h5
 ┣ 📄 evaluate_model.py
 ┣ 📄 model_testing.py
 ┣ 📄 README.md
 ┣ 📄 requirement.txt
 ┣ 📄 tempCodeRunnerFile.py
 ┣ 📄 train_vgg16_fer.py
```

## Model Training and Evaluation
1. **Pretrained Model**: Initially, `1stprj.py` was implemented using a pre-trained model to provide results.
2. **Training VGG16**: The model was later trained using the FER2013 dataset in `train_vgg16_fer.py`. The dataset is stored in the `dataset` directory.
3. **Model Evaluation**: The accuracy of the trained model is checked using `evaluate_model.py`, which reports approximately 47% accuracy, whereas during training, the model achieved up to 65% accuracy.
4. **Testing on Custom Images**: `model_testing.py` allows testing the trained model on custom images to observe its performance.

## Usage
1. Select an image to analyze.
2. View activation maps and feature maps.

## Example Output
The project will generate visualizations like:
- Feature maps showing different learned filters
- Heatmaps highlighting activated regions
- Side-by-side comparisons of different layers

## Contributing
Feel free to fork the repository and submit pull requests for improvements or new features.

## Acknowledgments
Special thanks to the open-source community and libraries that made this project possible.


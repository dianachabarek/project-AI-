
# Street Segmentation Project

This repository contains the code for a street segmentation project using convolutional neural networks (CNN) for image segmentation.

## Project Structure

- `pics/`: Contains the image data used for training and testing.
- `segm/`: Contains the segmentation masks corresponding to the images in `pics/`.
- `Street_segmentation_model.h5`: Saved model file after training.

## Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/dianachabarek/project-AI-.git 
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the image and segmentation data and place them in the respective folders (`pics/` and `segm/`).

## Usage

1. Train the model:

   ```bash
   python train_model.py
   ```

2. Make predictions:

   ```bash
   python predict.py <image_path>
   ```

## Model Architecture

The model architecture is based on a simple U-Net model for image segmentation. It consists of convolutional layers with relu activation, max-pooling layers, and transpose convolution layers for upsampling.

## Acknowledgements

- This project is based on the [Street Segmentation Dataset](https://www.cityscapes-dataset.com/login/).


## Author

- Diana Chabarek, Trey Shurley, Ryan Chin 
- Email: diana.chabarek@icloud.com , treyshurley@me.com, chinr1@my.erau.edu

---

Feel free to customize the sections and content based on your project specifics and requirements.
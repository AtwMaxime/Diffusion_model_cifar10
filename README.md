# Diffusion Model for CIFAR-10 Image Denoising

This project implements a **Diffusion Model** for denoising images, specifically for the **CIFAR-10** dataset. The model is built using a U-Net architecture with dropout layers and trained using a custom loss function. The loss function combines **L1 loss** and **SSIM loss** to improve the model's performance on image denoising tasks.

## Features

- **Image Denoising**: The model is trained to remove noise from CIFAR-10 images.
- **U-Net Architecture**: Uses a U-Net model with dropout for improved generalization.
- **Custom Loss Function**: Combines L1 loss with SSIM loss for better perceptual quality in denoised images.
- **Training and Validation**: Model training with early stopping and model checkpointing.
- **Noise Levels**: The model is evaluated at different noise levels (Gaussian noise) to assess its denoising performance.

## Prerequisites

Ensure you have the following Python libraries installed:

- Python 3.x
- PyTorch (with CUDA support)
- torchvision
- matplotlib
- numpy
- scikit-image

You can install the necessary dependencies using `pip`

## Dataset

The model is trained on the CIFAR-10 dataset, which can be downloaded automatically during the data loading process. CIFAR-10 contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Training

### Training Script

To train the model, run the following command:


python train.py


### Description

Model: U-Net with Dropout layers.
Loss Function: A custom loss function that combines L1 loss and SSIM loss.
Optimizer: Adam optimizer with weight decay.
Learning Rate Scheduler: StepLR with a step size of 10 and a gamma of 0.7.
Early Stopping: Training will stop if there is no improvement in validation loss for 3 consecutive epochs.

### Hyperparameters

batch_size: 64
learning_rate: 0.001
num_epochs: 30
weight_decay: 1e-4
patience: 3 (for early stopping)

## Testing

### Testing the Model

After training, you can test the model on a noisy image using the following command:

python generate.py --image_path path_to_image.png --model_path path_to_trained_model.pth

### Generate Random Image

You can also generate a new image from random noise using the trained model with the following command:

python generate.py --generate_random

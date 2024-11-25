import torch
from models.loss import DiffusionLoss
from data.preprocessing import load_data
from utils.noise import add_noise
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from models.unet import UNetWithDropout
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = UNetWithDropout(in_channels=3, out_channels=3).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = DiffusionLoss(ssim_weight=0.1)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

num_epochs = 30
patience = 3  # Early stopping patience (how many epochs to wait before stopping if no improvement)
best_val_loss = float('inf')
epochs_without_improvement = 0

def visualize_images(inputs, outputs, epoch):
    """Visualizes input images and their corresponding outputs every 3 epochs"""
    inputs = inputs.cpu().detach()
    outputs = outputs.cpu().detach()

    # Create grid of images
    input_grid = make_grid(inputs, nrow=8, normalize=True)
    output_grid = make_grid(outputs, nrow=8, normalize=True)

    # Plot images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(input_grid.permute(1, 2, 0))  # Input images
    plt.title(f'Epoch {epoch}: Inputs')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output_grid.permute(1, 2, 0))  # Output images
    plt.title(f'Epoch {epoch}: Outputs')
    plt.axis('off')

    plt.show()

def train():
    global epochs_without_improvement, best_val_loss
    model.train()
    print('Model set to training mode')

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Use custom loss (DiffusionLoss or other)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        # Print training loss
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}")

        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for i, (inputs, _) in enumerate(val_loader):
                inputs = inputs.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save model checkpoint
            torch.save(model.state_dict(), 'D:/Project VS/diffusion_model_cifar10/outputs/best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Visualize images every 3 epochs
        if epoch % 3 == 0:
            visualize_images(inputs, outputs, epoch)

if __name__ == "__main__":
    train()
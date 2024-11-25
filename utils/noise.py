import torch

def add_noise(images, noise_factor=0.1):
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return noisy_images
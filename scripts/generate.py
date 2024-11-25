import torch
from models.unet import UNetWithDropout
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise to an image."""
    noise = torch.randn_like(image) * noise_level
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)  # Ensure pixel values are in range [0, 1]

def denormalize(image, mean, std):
    """Denormalize the image from normalized [-1, 1] or [0, 1] range."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).cuda()
    std = torch.tensor(std).view(1, 3, 1, 1).cuda()
    return image * std + mean

def visualize_results(original, noisy, denoised):
    """Visualize original, noisy, and denoised images side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Original", "Noisy", "Denoised"]
    images = [original, noisy, denoised]

    for ax, img, title in zip(axes, images, titles):
        # Convert to numpy for visualization
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)  # Ensure valid range
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def calculate_metrics(original, denoised):
    """
    Calculate PSNR and SSIM metrics between two images.
    """
    # Remove batch dimension if present
    if original.ndim == 4:
        original = original.squeeze(0)  # From (1, C, H, W) to (C, H, W)
    if denoised.ndim == 4:
        denoised = denoised.squeeze(0)

    # Convert tensors to NumPy arrays and transpose to HWC format
    original_np = original.cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
    denoised_np = denoised.cpu().numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C

    # PSNR
    psnr_value = psnr(original_np, denoised_np, data_range=1)  # Assuming normalized [0,1] images

    # SSIM with an adjusted window size
    ssim_value = ssim(
        original_np,
        denoised_np,
        win_size=3,  # Set an appropriate odd value <= 7 (e.g., 3 or 5 for CIFAR-10 images)
        channel_axis=-1,
        data_range=1,
    )
    return psnr_value, ssim_value

def test_model_with_image(image_path, model_path="outputs/best_model.pth", noise_levels=[0.1, 0.2, 0.5]):
    """Test the diffusion model with a specific image at different noise levels."""
    # Load the trained model
    model = UNetWithDropout(in_channels=3, out_channels=3).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    original_image = transform(image).unsqueeze(0).cuda()

    for noise_level in noise_levels:
        print(f"\nTesting with noise level: {noise_level}")
        noisy_image = add_gaussian_noise(original_image, noise_level)

        with torch.no_grad():
            denoised_image = model(noisy_image)

        psnr_value, ssim_value = calculate_metrics(original_image, denoised_image)
        print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

        # Visualize the results
        visualize_results(
            original_image.squeeze(0),
            noisy_image.squeeze(0),
            denoised_image.squeeze(0)
        )

def generate_random_image(model_path="outputs/best_model.pth"):
    """Generate a new image starting from random noise."""
    # Load the trained model
    model = UNetWithDropout(in_channels=3, out_channels=3).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate a random noise input
    random_noise = torch.randn(1, 3, 32, 32).cuda()

    with torch.no_grad():
        generated_image = model(random_noise)

    # Convert tensor to image and display
    generated_image = generated_image.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(np.clip(generated_image, 0, 1))
    plt.title("Generated Image")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Test the model with a specific image
    test_image_path = "D:/Project VS/diffusion_model_cifar10/dog32x32.png"  # Replace with the path to a test image
    test_model_with_image(test_image_path, model_path="outputs/best_model.pth")

    # Generate a random new image
    generate_random_image(model_path="outputs/best_model.pth")

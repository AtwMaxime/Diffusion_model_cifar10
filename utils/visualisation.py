import matplotlib.pyplot as plt

def plot_image(image_tensor):
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.show()

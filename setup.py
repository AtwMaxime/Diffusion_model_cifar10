
from setuptools import setup, find_packages

setup(
    name='diffusion_model_cifar10',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'tqdm',
        'numpy'
    ],
    description='Diffusion model implementation for CIFAR-10 using PyTorch',
    author='Your Name',
    license='MIT'
)


import torch
from torchvision import datasets, transforms

def get_colored_mnist(data_dir='data/raw', train=True, download=True):
    # Use torchvision's MNIST, then colorize images as per paper's protocol
    mnist = datasets.MNIST(
        data_dir, train=train, download=download,
        transform=transforms.ToTensor()
    )
    # TODO: Apply colorization and return colored dataset
    return mnist

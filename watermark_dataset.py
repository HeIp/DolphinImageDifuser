from PIL import Image #PIL.Image: This library is used for image processing tasks like opening and converting images.
import os #This library provides functionalities for interacting with the operating system, such as listing files in a directory.
import torch #The core PyTorch library for deep learning.
from torchvision import models #Provides pre-trained models, datasets, and image transformation functions.
from torchvision import transforms # Offers functions for transforming images (e.g., resizing, normalization).
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
#This class serves as the base class for creating custom datasets.
#DataLoader: This class helps manage loading batches of data during training and evaluation.




#Dataset from torchvision.utils.data
# It's designed to load and pre-process watermark images for training a deep learning mode
class WatermarkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if
                            f.lower().endswith(('.jpg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image
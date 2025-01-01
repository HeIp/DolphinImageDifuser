from PIL import Image #For image processing tasks.
import os #For file system interactions (e.g., listing files in a directory).
import torch #The core PyTorch library for deep learning.
from torchvision import models #Provides pre-trained models, datasets, and image transformation functions.
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import cv2 # OpenCV library for computer vision tasks (e.g., image manipulation, color space conversion).
import numpy as np

# Load the trained model
model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Load pre-trained ResNet-50
num_classes = 2  # Binary classification: with watermark, without watermark
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Replace the final fully connected layer
model.load_state_dict(torch.load("watermark_detector_weights.pth", weights_only=True))  # Load trained weights
model.eval()
print('Loaded Model saved')

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

def predict_watermark(image_path, model, transform):
    """
    Predicts whether an image contains a watermark using the trained model.

    Args:
        image_path: Path to the image.
        model: Trained PyTorch model.
        transform: Image transformations applied during training.

    Returns:
        True if the model predicts a watermark, False otherwise.
    """
    print('Predict water mark method started')
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image)

    _, predicted = torch.max(output.data, 1)
    if predicted.item() == 1:
        print(f"Watermark detected in {image_path}")  # Print if watermark is detected
    return predicted.item() == 1  # 1 represents "with watermark"


def create_color_based_mask(image_path, lower_bound, upper_bound):
    """
    Creates a mask for watermark removal based on color differences.

    Args:
        image_path: Path to the image.
        lower_bound: Lower bound for color range in HSV format (e.g., [0, 100, 100]).
        upper_bound: Upper bound for color range in HSV format (e.g., [20, 255, 255]).

    Returns:
        Mask image (binary image where watermark pixels are 255 and background is 0).
    """
    print('Create color mask started')
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

    # Refine the mask using morphological operations (optional)
    kernel = np.ones((3, 3), np.uint8)  # Define kernel for morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Open operation to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close operation to fill small holes

    # Debug: Save the mask to check if it's created correctly
    mask_path = image_path.replace(".jpg", "_mask.jpg")
    cv2.imwrite(mask_path, mask)
    print(f"Mask saved to {mask_path}")

    return mask


def remove_watermark_inpaint(image_path, mask):
    """
    Removes the watermark from an image using OpenCV's inpainting function.

    Args:
        image_path: Path to the image.
        mask: Mask image indicating the region of the watermark (0 for background, 255 for watermark).

    Returns:
        Path to the modified image, or None if inpainting fails.
    """
    print(f"Starting inpainting for {image_path}")

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        img_no_watermark = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)  # Use Inpaint_TELEA algorithm
    except cv2.error:
        print(f"Inpainting failed for {image_path}")
        return None

    modified_path = image_path.replace(".jpg", "_nowatermark.jpg")  # Save with "_nowatermark" suffix
    cv2.imwrite(modified_path, cv2.cvtColor(img_no_watermark, cv2.COLOR_RGB2BGR))

    print(f"Inpainting completed for {image_path}, saved to {modified_path}")

    return modified_path


# Remove watermarks from images in ./SealWaterMark
watermark_folder = "./SealWaterMark"
for filename in os.listdir(watermark_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(watermark_folder, filename)
        print(f"Checking image in {image_path}")
        if predict_watermark(image_path, model, transform):  # Predict watermark presence
            print(f"Watermark detected in {image_path}")

            # Improved mask creation (replace with your preferred method)
            # Here's a placeholder for color-based segmentation (replace with your implementation)
            lower_bound = np.array([0, 100, 100])  # Example lower bound for HSV
            upper_bound = np.array([20, 255, 255])  # Example upper bound for HSV
            mask = create_color_based_mask(image_path, lower_bound, upper_bound)
            print("mask created")


            if mask is not None and np.any(mask):
                modified_path = remove_watermark_inpaint(image_path, mask)
                if modified_path:
                    print(f"Watermark removed and saved to {modified_path}")
                else:
                    print(f"Inpainting failed for {image_path}")
            else:
                print(f"Mask creation failed for {image_path}")
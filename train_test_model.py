import torch #The core PyTorch library for deep learning.
from torchvision import models #Provides pre-trained models, datasets, and image transformation functions.
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from watermark_dataset import WatermarkDataset  # Import the WatermarkDataset class

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load datasets
train_dataset = WatermarkDataset(root_dir="./YesWaterMark", transform=transform)
test_dataset = WatermarkDataset(root_dir="./NoWaterMark", transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=66, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=66, shuffle=False)

# Load pre-trained ResNet-50 model (using recommended weights)
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Use pretrained=True for default weights

# Freeze convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer with a new layer
num_classes = 2  # Binary classification: with watermark, without watermark
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model
num_epochs = 66
for epoch in range(num_epochs):
    model.train()
    for i, images in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        labels = torch.zeros(images.shape[0]).long()  # Assuming "YesWaterMark" is class 1
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += images.size(0)
        correct += (predicted == torch.zeros(images.shape[0]).long()).sum().item()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))

# Save the trained model
torch.save(model.state_dict(), "watermark_detector_weights.pth")  # Save only model weights
print('Model saved')
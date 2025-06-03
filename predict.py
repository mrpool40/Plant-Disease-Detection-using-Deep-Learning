import torch
from torchvision import transforms
from PIL import Image
import sys

from models.model import get_model
from utils.data_loader import get_data_loaders

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'dataset/plantvillage'
_, _, _, class_names = get_data_loaders(data_dir)
num_classes = len(class_names)

# Load model
model = get_model(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load image path from command line argument
if len(sys.argv) != 2:
    print("Usage: python predict.py path_to_image.jpg")
    sys.exit()

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    prediction = class_names[predicted.item()]

print(f"\n✅ Predicted Class: {prediction}")

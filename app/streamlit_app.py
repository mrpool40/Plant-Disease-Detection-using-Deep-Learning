import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# 🔧 Fix import path to access models/ and utils/ folders
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model import get_model
from utils.data_loader import get_data_loaders

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names and model
data_dir = 'dataset/plantvillage'
_, _, _, class_names = get_data_loaders(data_dir)
num_classes = len(class_names)

model = get_model(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Classifier")
st.title("🌿 Plant Disease Detection App")
st.markdown("Upload a leaf image to detect the disease category.")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        prediction = class_names[pred.item()]

    st.success(f"✅ Prediction: **{prediction}**")

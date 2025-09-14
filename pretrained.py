# app_plant_streamlit.py
import io
import os
import requests
import numpy as np
import cv2
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "plant_model.pth"   # change to your model path or preload a HF link
CLASS_NAMES = [
    # Replace with your actual class names in the same order the model was trained
    "Apple___healthy", "Apple___scab", "Apple___black_rot", "Apple___rust",
    # ... add the rest from your dataset
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utility functions
# -------------------------
@st.cache_resource
def load_model(path=MODEL_PATH, num_classes=None):
    """Load a PyTorch ResNet18 and attach saved weights (state_dict)."""
    if num_classes is None:
        num_classes = len(CLASS_NAMES)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            state = torch.load(path, map_location=DEVICE)
            # state may be either state_dict or a checkpoint dict
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.eval()
            st.info(f"Loaded model from {path}")
        except Exception as e:
            st.warning(f"Failed to load weights from {path}: {e}")
    else:
        st.warning(f"No model file found at {path}. App will still run but predictions will be random.")
    return model

def download_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def read_image_from_upload(uploaded_file):
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def leaf_detection_bbox(img_bgr):
    """
    Simple color-based leaf segmentation + largest contour bounding box.
    Returns bbox as (x,y,w,h) or None if not found.
    """
    if img_bgr is None:
        return None
    # Convert to HSV and threshold for green-ish colors (tweak ranges if needed)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # HSV range for leaves — adjust if your leaves look different
    lower = np.array([20, 30, 20])   # H,S,V
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and pick the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:  # too small -> ignore
        return None
    x,y,w,h = cv2.boundingRect(largest)
    return (x,y,w,h), mask

def preprocess_for_model(crop_bgr, img_size=224):
    """Preprocess crop for ResNet: BGR->RGB, resize, normalize, to tensor."""
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = Image.fromarray(img)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    x = tf(img).unsqueeze(0).to(DEVICE)  # 1,C,H,W
    return x

def predict(model, crop_bgr, topk=3):
    x = preprocess_for_model(crop_bgr)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        topk_idx = probs.argsort()[::-1][:topk]
        results = [(CLASS_NAMES[i], float(probs[i])) for i in topk_idx]
    return results, probs

# -------------------------
# Main UI
# -------------------------
st.title("🌿 Plant Disease Detection (ResNet18)")

st.markdown("""
Upload a leaf image or paste an image URL. The app will try to detect the leaf, draw a bounding box,
crop the leaf region and predict the disease using a pretrained ResNet18 model.
""")

col1, col2 = st.columns([3,1])

with col1:
    url = st.text_input("Image URL (http/https)", "")
    uploaded_file = st.file_uploader("Or upload an image file", type=["jpg","jpeg","png"])

with col2:
    st.write("Model")
    st.write(f"Device: **{DEVICE}**")
    model_path = st.text_input("Model path (optional)", MODEL_PATH)
    if st.button("Reload model"):
        # reload by clearing cache resource (Streamlit caches persist until code changes)
        load_model.cache_clear()
        model = load_model(model_path, num_classes=len(CLASS_NAMES))
    else:
        model = load_model(model_path, num_classes=len(CLASS_NAMES))

# Load image
img = None
if url:
    try:
        img = download_image_from_url(url)
    except Exception as e:
        st.error(f"Failed to download image: {e}")
elif uploaded_file is not None:
    try:
        img = read_image_from_upload(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")

if img is None:
    st.info("No image loaded yet. Paste a URL or upload one.")
    st.stop()

# Resize preview for display (keep original for crop)
preview = img.copy()
preview_h, preview_w = preview.shape[:2]
display_max = 800
if max(preview_h, preview_w) > display_max:
    scale = display_max / max(preview_h, preview_w)
    preview = cv2.resize(preview, (int(preview_w*scale), int(preview_h*scale)))

# Detect leaf bounding box
det = leaf_detection_bbox(img)
if det is None:
    st.warning("No leaf detected using the simple color segmentation. The app will predict on the full image.")
    crop = img.copy()
    bbox = None
else:
    (x,y,w,h), mask = det
    crop = img[y:y+h, x:x+w]
    bbox = (x,y,w,h)

# Run prediction
results, probs = predict(model, crop, topk=5)

# Annotate image (draw bbox + top label)
annot = preview.copy()
if bbox is not None:
    # scale bbox to preview size if preview was resized
    scale_x = preview.shape[1] / img.shape[1]
    scale_y = preview.shape[0] / img.shape[0]
    x_s = int(x * scale_x); y_s = int(y * scale_y)
    w_s = int(w * scale_x); h_s = int(h * scale_y)
    cv2.rectangle(annot, (x_s, y_s), (x_s+w_s, y_s+h_s), (0,255,0), 2)
    label_text = f"{results[0][0]} ({results[0][1]*100:.1f}%)"
    cv2.putText(annot, label_text, (x_s, max(0,y_s-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
else:
    # whole-image label
    label_text = f"{results[0][0]} ({results[0][1]*100:.1f}%)"
    cv2.putText(annot, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

# Display annotated image
st.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB), caption="Annotated image", use_column_width=True)

# Show results table
st.subheader("Top predictions")
for name, p in results[:5]:
    st.write(f"- **{name}** — {p*100:.2f}%")

# Optionally show the segmentation mask
if bbox is not None and st.checkbox("Show segmentation mask"):
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlaid = cv2.addWeighted(img, 0.6, mask_vis, 0.4, 0)
    st.image(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB), caption="Segmentation overlay", use_column_width=True)

st.markdown("---")
st.caption("Notes: Adjust leaf_detection_bbox() color ranges if your dataset's leaves are different in color. "
           "Replace CLASS_NAMES and MODEL_PATH with your actual classes and model file.")

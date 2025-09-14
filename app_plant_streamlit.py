# Colab inline Plant Disease Detection dashboard (PyTorch ResNet18)
# Paste & run this single cell in Colab.

# 0) Install required packages (first run)
!pip install -q ipywidgets matplotlib opencv-python-headless torch torchvision pillow requests

# 1) Imports
import os, io, requests, time
from pathlib import Path
from IPython.display import display, clear_output
import ipywidgets as widgets
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 2) Configuration - adjust these if needed
MODEL_PATH = "best_model.pth"   # local checkpoint filename (state_dict or checkpoint)
CLASS_NAMES = [                   # placeholder: replace with your actual class names in model order
    "Apple___healthy", "Apple___scab", "Apple___black_rot", "Apple___rust"
    # add all classes here...
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_MAX = 800  # scale large images for display

# 3) Model loading utility
def load_resnet18_model(model_path=MODEL_PATH, num_classes=None, device=DEVICE):
    if num_classes is None:
        num_classes = len(CLASS_NAMES)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    model.eval()
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            ckpt = torch.load(model_path, map_location=device)
            # support both state_dict and checkpoint dict containing 'model_state_dict'
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif isinstance(ckpt, dict) and all(k.startswith("module.") or k in model.state_dict() for k in ckpt.keys()):
                # might already be state_dict
                state = ckpt
            else:
                # try direct as state_dict
                state = ckpt
            model.load_state_dict(state)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print("Warning: failed to load model weights:", e)
            print("Model architecture created but weights not loaded.")
    else:
        print(f"No model file found at {model_path}. Model created with random weights. Place your .pth at this path to use trained weights.")
    return model

# instantiate model (may take a moment)
print("Initializing model (this may take a few seconds)... Device:", DEVICE)
model = load_resnet18_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=DEVICE)

# 4) Image utilities
def download_image_from_url(url, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def read_image_from_upload(uploaded):
    # uploaded is ipywidgets FileUpload .value -> dict of {filename: {'content': bytes, ...}}
    if not uploaded:
        return None
    # handle both dict and list-like structures
    if isinstance(uploaded, dict):
        first = next(iter(uploaded.values()))
        b = first['content']
    else:
        # list-like
        first = uploaded[0]
        if isinstance(first, dict) and 'content' in first:
            b = first['content']
        else:
            # maybe object with .data
            b = first
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def scale_for_display(img, max_side=DISPLAY_MAX):
    h,w = img.shape[:2]
    if max(h,w) <= max_side:
        return img, 1.0
    scale = max_side / max(h,w)
    neww = int(w*scale); newh = int(h*scale)
    return cv2.resize(img, (neww,newh)), scale

# 5) Leaf detection: simple HSV-based segmentation -> largest contour bbox
def leaf_detection_bbox(img_bgr):
    if img_bgr is None:
        return None, None
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # HSV range for green (tweak if your leaves differ)
    lower = np.array([20, 30, 20])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:  # ignore tiny
        return None, mask
    x,y,w,h = cv2.boundingRect(largest)
    return (x,y,w,h), mask

# 6) Preprocess crop for PyTorch ResNet
tf_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_crop_for_model(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = tf_preprocess(rgb).unsqueeze(0).to(DEVICE)
    return tensor

def predict_model_on_crop(crop_bgr, topk=5):
    if crop_bgr is None:
        return []
    x = preprocess_crop_for_model(crop_bgr)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:topk]
    return [(CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), float(probs[i])) for i in top_idx], probs

# 7) Widgets: URL box, file upload, button and output area
url_box = widgets.Text(value='', placeholder='Paste image URL (http/https) here', description='Image URL:', layout=widgets.Layout(width='70%'))
file_uploader = widgets.FileUpload(accept='image/*', multiple=False)
run_button = widgets.Button(description='Detect & Predict', button_style='primary')
output = widgets.Output(layout=widgets.Layout(border='1px solid #ccc', padding='10px'))

# Helper: read uploaded file value (ipywidgets returns dict in Jupyter, in Colab it behaves similarly)
def get_uploaded_value(uploader_widget):
    # uploader_widget.value is a dict in classic ipywidgets: filename -> metadata
    try:
        val = uploader_widget.value
        if not val:
            return None
        # If it's dict-like
        if isinstance(val, dict):
            first = next(iter(val.values()))
            return first['content']
        # If it's list-like
        if isinstance(val, (list, tuple)):
            # each item may be dict
            first = val[0]
            if isinstance(first, dict) and 'content' in first:
                return first['content']
            # maybe it's an UploadedFile object - read .data?
            return first
    except Exception:
        return None

# 8) Main action handler
def on_run_clicked(b):
    with output:
        clear_output(wait=True)
        print("Processing... (this may take a few seconds)")
        # 1) load image (URL takes precedence)
        img = None
        if url_box.value.strip():
            try:
                img = download_image_from_url(url_box.value.strip())
            except Exception as e:
                print("Failed to download URL:", e)
                img = None
        if img is None and file_uploader.value:
            try:
                content = get_uploaded_value(file_uploader)
                if content is None:
                    # older API: file_uploader.value is a list-like with dicts
                    content = list(file_uploader.value)[0]['content'] if file_uploader.value else None
                if content is None:
                    print("Unable to read uploaded file from widget.")
                    return
                arr = np.frombuffer(content, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                print("Failed to read uploaded file:", e)
                img = None

        if img is None:
            print("No image provided. Paste an URL or upload an image.")
            return

        # 2) detect leaf bbox
        bbox, mask = leaf_detection_bbox(img)
        if bbox is None:
            print("No leaf detected with simple segmentation. Predicting on full image.")
            crop = img.copy()
        else:
            x,y,w,h = bbox
            crop = img[y:y+h, x:x+w]

        # 3) predict
        try:
            results, probs = predict_model_on_crop(crop, topk=5)
        except Exception as e:
            print("Prediction failed:", e)
            return

        # 4) prepare annotated display image (scaled)
        disp_img, scale = scale_for_display(img, max_side=DISPLAY_MAX)
        annotated = disp_img.copy()
        if bbox is not None:
            x_s = int(x*scale); y_s = int(y*scale); w_s = int(w*scale); h_s = int(h*scale)
            cv2.rectangle(annotated, (x_s,y_s), (x_s+w_s, y_s+h_s), (0,255,0), 2)
            label_text = f"{results[0][0]} ({results[0][1]*100:.1f}%)"
            cv2.putText(annotated, label_text, (x_s, max(0,y_s-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        else:
            label_text = f"{results[0][0]} ({results[0][1]*100:.1f}%)"
            cv2.putText(annotated, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # 5) show annotated image inline
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.show()

        # 6) print results table
        print("Top predictions:")
        for i,(name, p) in enumerate(results):
            print(f" {i+1}. {name} — {p*100:.2f}%")

        # 7) optionally show mask
        if bbox is not None:
            # show small mask visualization
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            over = cv2.addWeighted(cv2.resize(img, (mask_rgb.shape[1], mask_rgb.shape[0])), 0.6, mask_rgb, 0.4, 0)
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
            ax2.axis('off')
            plt.title("Segmentation overlay (for debug)")
            plt.show()

# bind button
run_button.on_click(on_run_clicked)

# 9) Layout & display
title = widgets.HTML("<h3>Plant Disease Detection — Colab inline dashboard</h3><p>Paste an image URL or upload a file, then click <b>Detect & Predict</b>.</p>")
controls = widgets.HBox([url_box, run_button])
uploader_row = widgets.HBox([widgets.Label("Or upload:"), file_uploader])
display(title, controls, uploader_row, output)

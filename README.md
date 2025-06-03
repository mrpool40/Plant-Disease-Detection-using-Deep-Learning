
# 🌿 Plant Disease Detection using Deep Learning

This project is a deep learning image classification system for detecting plant diseases from leaf images using PyTorch and ResNet18. It features a **Streamlit-powered web app** that allows users to upload a leaf image and receive an instant disease prediction.

---

## 🧠 Model Overview

- **Architecture**: Transfer learning using `ResNet18`
- **Dataset**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Frameworks**: PyTorch, torchvision
- **App**: Built with Streamlit for interactive use
- **Performance**: Achieved high accuracy using a custom split of train/val/test sets with data augmentation

---

## 🗂 Project Structure

```
plant-disease-detection/
├── app/
│   └── streamlit_app.py       # Streamlit web app
├── models/
│   └── model.py               # ResNet18 model setup
├── utils/
│   └── data_loader.py         # Data loading and augmentation
├── dataset/
│   └── plantvillage/          # train/val/test image splits
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── predict.py                 # Predict from a single image
├── split_dataset.py           # Split raw dataset into train/val/test
├── best_model.pth             # Saved trained model weights
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
```

### 2️⃣ Create and activate virtual environment
```bash
python -m venv plant_env
plant_env\Scripts\activate  # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, install manually:

```bash
pip install torch torchvision pandas matplotlib seaborn opencv-python streamlit scikit-learn tqdm
```

---

## 📦 Dataset

- Download the PlantVillage dataset from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Extract the dataset into `dataset/PlantVillage`
- Split the dataset using:
```bash
python utils/split_dataset.py
```

---

## 🏋️‍♀️ Train the Model

```bash
python train.py
```

The best model will be saved as `best_model.pth`.

---

## 📊 Evaluate the Model

```bash
python evaluate.py
```

Outputs classification report and a confusion matrix heatmap.

---

## 🖼 Predict a Single Image

```bash
python predict.py "path_to_image.jpg"
```

---

## 🌐 Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

Then open: `http://localhost:8501` to upload and classify leaf images.

---

## 📷 Sample

![App Screenshot](https://via.placeholder.com/600x400.png?text=App+Screenshot+Placeholder)

---

## 💡 Future Improvements

- Support for mobile/webcam input
- Confidence thresholds and top-k predictions
- Deployment to Streamlit Cloud / Hugging Face Spaces
- Model improvements using EfficientNet, MobileNet, or ViT

---


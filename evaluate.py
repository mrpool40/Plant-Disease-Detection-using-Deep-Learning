import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_loader import get_data_loaders
from models.model import get_model

# Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_dir = 'dataset/plantvillage'
_, _, test_loader, class_names = get_data_loaders(data_dir)
num_classes = len(class_names)

# Load model
model = get_model(num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

# Evaluate
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

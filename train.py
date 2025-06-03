import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils.data_loader import get_data_loaders
from models.model import get_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATA_DIR = 'dataset/plantvillage'
SAVE_PATH = 'best_model.pth'
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Load data
train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
num_classes = len(class_names)

# Load model
model = get_model(num_classes)
model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # --- Training ---
    model.train()
    train_loss, train_correct = 0.0, 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    train_acc = train_correct / len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    val_loss /= len(val_loader.dataset)

    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved with val accuracy: {val_acc:.4f}")

print("Training complete.")

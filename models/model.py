import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze the earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model

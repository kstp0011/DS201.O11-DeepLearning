from torchvision import models
import torch.nn as nn
import torch

def DefineModel_ResNet50(class_names):
    model_resnet50 = models.resnet50(weights='IMAGENET1K_V2')

    # Freeze training for all "features" layers
    for param in model_resnet50.parameters():
        param.requires_grad = False

    num_features = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_features, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_resnet50 = model_resnet50.to(device)

    return model_resnet50
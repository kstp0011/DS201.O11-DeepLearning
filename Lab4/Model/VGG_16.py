from torchvision import models
import torch.nn as nn
import torch

def DefineModel_VGG16(class_names):
    model = models.vgg16(weights='IMAGENET1K_V1')
    # Freeze training for all layers
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(class_names))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

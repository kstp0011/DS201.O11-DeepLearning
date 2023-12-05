from torchvision import models
import torch.nn as nn
import torch

def DefineModel_VGG19(class_names):
    model_vgg19 = models.vgg19(weights='IMAGENET1K_V1')

    # Freeze training for all "features" layers
    for param in model_vgg19.features.parameters():
        param.requires_grad = False

    num_features = model_vgg19.classifier[6].in_features
    model_vgg19.classifier[6] = nn.Linear(num_features, 2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_vgg19 = model_vgg19.to(device)
    return model_vgg19
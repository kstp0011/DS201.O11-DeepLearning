from DataLoader import Chest_Xray_dataloaders, Chest_Xray_class_names, Chest_Xray_dataset_sizes
from Utils import train_model, evaluate_model
from Model import DefineModel_ResNet50
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn



def main():
    model = DefineModel_ResNet50(Chest_Xray_class_names)
    print(model)

    print("Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, Chest_Xray_dataloaders, Chest_Xray_dataset_sizes, num_epochs=10)

    print("Evaluating...")
    evaluate_model(model, Chest_Xray_dataloaders, Chest_Xray_class_names)

if __name__ == "__main__":
    main()
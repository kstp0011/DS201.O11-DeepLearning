from DataLoader import Chest_Xray_dataloaders, Chest_Xray_class_names, Chest_Xray_dataset_sizes
from Utils import train_model, evaluate_model
from Model import DefineModel_VGG16

def main():
    model = DefineModel_VGG16(Chest_Xray_class_names)
    print(model)

    print("Training...")
    train_model(model, Chest_Xray_dataloaders, Chest_Xray_dataset_sizes, Chest_Xray_class_names, num_epochs=25)
    
    print("Evaluating...")
    evaluate_model(model, Chest_Xray_dataloaders, Chest_Xray_dataset_sizes, Chest_Xray_class_names)

if __name__ == "__main__":
    main()
# Thanks to Tim Dangeon on Kaggle --> https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/discussion/482896 for the remove duplicate images code.

import os
import hashlib
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
#from models import TransferLearningResNet  # Import your model architecture here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to compute hash for detecting duplicates
def compute_hash(file):
    hasher = hashlib.md5()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


# Function to list files and detect duplicates
def list_files(hash_dict, dataset_path):
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if file.endswith(".jpg"):
                file_hash = compute_hash(file_path)
                if file_hash in hash_dict:
                    hash_dict[file_hash].append(file_path)
                else:
                    hash_dict[file_hash] = [file_path]


# Function to remove duplicate images
def remove_duplicates(hash_dict):
    duplicate_count = 0
    for hash_value, file_paths in hash_dict.items():
        if len(file_paths) > 1:
            for file_path in file_paths[1:]:
                print(f"Removing duplicate (hash: {hash_value}): {file_path}")
                os.remove(file_path)
                duplicate_count += 1
    print(f"Number of duplicates removed: {duplicate_count}")


# Function to visualize a few images from the dataset
def visualize_data_one_per_class(dataset):
    class_indices = {class_name: None for class_name in dataset.classes}  # Dictionary to store indices of one image per class

    # Loop over the dataset to find one image for each class
    for idx, (image, label) in enumerate(dataset):
        class_name = dataset.classes[label]
        if class_indices[class_name] is None:  # If we haven't picked an image for this class yet
            class_indices[class_name] = idx
        if all(index is not None for index in class_indices.values()):  # Stop once we have one image for each class
            break

    # Plot one image per class
    fig, axes = plt.subplots(1, len(class_indices), figsize=(15, 5))

    for i, (class_name, idx) in enumerate(class_indices.items()):
        image, label = dataset[idx]

        if isinstance(image, torch.Tensor):
            image_np = image.numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        else:
            image_np = np.array(image)  # Convert PIL image to NumPy array

        # Denormalize the image if needed (comment out if not using normalization)
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)  # Clip values to valid range [0, 1]

        axes[i].imshow(image_np)
        axes[i].axis('off')
        axes[i].set_title(f'Class: {class_name}')

    plt.show()


# Function to evaluate model and generate confusion matrix
def evaluate_model(model, loader, class_names):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def main():
    # Define the transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increase rotation angle
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Add random cropping
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),  # Aggressive color jitter
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets with transformations
    train_dataset_path = "data/seg_train/seg_train"
    valid_dataset_path = "data/seg_test/seg_test"

    # Load training dataset with transformations
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Classes in training dataset: {train_dataset.classes}")  # Check the classes

    # Load validation dataset with transformations
    valid_dataset = datasets.ImageFolder(root=valid_dataset_path, transform=valid_transforms)
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Classes in validation dataset: {valid_dataset.classes}")  # Check the classes

    # Visualize 5 images from the training dataset
    print("Visualizing 5 images from the training dataset:")
    visualize_data_one_per_class(train_dataset)

    # Visualize 5 images from the validation dataset
    print("Visualizing 5 images from the validation dataset:")
    visualize_data_one_per_class(valid_dataset)

    # Load model (make sure you have the trained model saved as 'resnet_model.pth')
    from models import TransferLearningResNet  # Import your model class here
    model_path = "./resnet_model.pth"
    model = TransferLearningResNet(num_classes=len(train_dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create a DataLoader for validation dataset
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    print("Evaluating the model on the validation dataset:")
    evaluate_model(model, valid_loader, class_names=valid_dataset.classes)

if __name__ == "__main__":
    main()
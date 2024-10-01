import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from models import TransferLearningResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model
def load_model(model_path, num_classes=6):
    model = TransferLearningResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


# Define the image transforms (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels (make sure these match your dataset's class names)
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


# Function to predict on a single image
def predict_image(model, image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.item(), class_names[predicted.item()], image  # Return class index, class name, and original image


# Function to display image and prediction
def display_prediction(image, prediction_label):
    plt.imshow(image)
    plt.title(f'Predicted: {prediction_label}')
    plt.axis('off')
    plt.show()


# Main function to load images, predict, and display results
def predict_on_folder(model, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)

        # Get prediction
        class_idx, class_name, image = predict_image(model, image_path)

        # Display the result
        print(f'Image: {image_file}, Predicted class: {class_name}')
        display_prediction(image, class_name)


if __name__ == "__main__":
    # Load the model
    model_path = "./resnet_model.pth"  # Path to your saved model
    model = load_model(model_path)

    # Path to the folder containing images for prediction
    pred_folder_path = "data/seg_pred/seg_pred"

    # Predict on the folder and display results
    predict_on_folder(model, pred_folder_path)

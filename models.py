from torch import nn
from torchvision import models

class TransferLearningResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate_1=0.1, dropout_rate_2=0.1):
        super(TransferLearningResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freeze all layers except the last block (layer4)
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze all layers

        # Unfreeze more layers for fine-tuning
        for param in self.resnet.layer2.parameters():
            param.requires_grad = True

        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        # Unfreeze layer4 for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Modify the final fully connected layers
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),  # Dense layer with 128 units
            nn.ReLU(),  # ReLU activation
            nn.Dropout(dropout_rate_1),  # Second dropout layer
            nn.Linear(512, 256),  # Another layer with more units
            nn.ReLU(),
            nn.Dropout(dropout_rate_2),
            nn.Linear(256, num_classes),  # Final dense layer with output matching number of classes
        )

    def forward(self, x):
        return self.resnet(x)


# Model factory updated to include both CNN and transfer learning options
model_factory = {
    'resnet': TransferLearningResNet,
}


# Option to save the model (unchanged from original)
def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            save(model.state_dict(), path.join('./', f"{n}_model.pth"))
            print(f"Model saved as {n}_model.pth")

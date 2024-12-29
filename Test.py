import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix


def preprocess_image(image):
    image = image.convert("L")
    image = np.array(image)
    local_mean = np.mean(image)
    thresh_image = np.where(image > (local_mean - 2), 255, 0)
    return Image.fromarray(thresh_image.astype(np.uint8))


class GlomeruliDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        image = preprocess_image(image)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return image, y_label


class GlomeruliClassifier(nn.Module):
    def __init__(self, base_model, num_classes=2):
        super(GlomeruliClassifier, self).__init__()
        self.base_model = base_model
        num_features = 2048
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, predictions, true_labels


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test data
    test_dataset = GlomeruliDataset(
        csv_file="./test_labels.csv",  # Path to your test labels CSV
        img_dir="./ResizedTestSet",  # Path to your test images directory
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Create a models directory in your project
    model_path = "./inception_v3_google-0cc3c7bd.pth"

    # Load the model with custom weights using safe loading
    base_model = models.inception_v3(weights=None, init_weights=True)
    base_model.load_state_dict(torch.load(model_path, weights_only=True))
    base_model.fc = nn.Identity()
    model = GlomeruliClassifier(base_model).to(device)

    # Load the trained model
    model.load_state_dict(torch.load("./Final_model.pth"))

    # Evaluate the model
    accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))

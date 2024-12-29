import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device
):
    best_loss = float("inf")
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{running_loss/(progress_bar.n+1):.4f}",
                    "LR": f'{optimizer.param_groups[0]["lr"]:.6f}',
                }
            )

        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * correct / total

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "Final_model.pth")
            print(f"Saved new best model with validation loss: {best_loss:.4f}")

        scheduler.step(val_loss)
        print("\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = GlomeruliDataset(
        csv_file="./train_labels.csv",
        img_dir="./ResizedTrainingSet",
        transform=transform,
    )

    labels = [int(dataset.annotations.iloc[i, 1]) for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print("Class distribution:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    model_path = "./inception_v3_google-0cc3c7bd.pth"
    base_model = models.inception_v3(pretrained=False)
    base_model.load_state_dict(torch.load(model_path))
    base_model.aux_logits = False
    base_model.fc = nn.Identity()
    model = GlomeruliClassifier(base_model).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=20,
        device=device,
    )

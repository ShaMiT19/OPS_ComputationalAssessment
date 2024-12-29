import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import csv
from tqdm import tqdm


def preprocess_image(image):
    image = image.convert("L")
    image = np.array(image)
    local_mean = np.mean(image)
    thresh_image = np.where(image > (local_mean - 2), 255, 0)
    return Image.fromarray(thresh_image.astype(np.uint8))


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


def predict_images(model, image_folder, device):
    model.eval()
    predictions = []
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with torch.no_grad():
        for image_name in tqdm(os.listdir(image_folder)):
            if image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_folder, image_name)
                image = Image.open(img_path)
                image = preprocess_image(image)
                image = image.convert("RGB")
                image = transform(image).unsqueeze(0).to(device)

                outputs = model(image)
                _, predicted = outputs.max(1)
                predictions.append((image_name, predicted.item()))

    return predictions


def save_predictions_to_csv(predictions, output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Name", "Prediction"])
        writer.writerows(predictions)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_path = "./inception_v3_google-0cc3c7bd.pth"
    base_model = models.inception_v3(weights=None, init_weights=True)
    base_model.load_state_dict(torch.load(model_path, weights_only=True))
    base_model.fc = nn.Identity()
    model = GlomeruliClassifier(base_model).to(device)

    # Load the trained model weights
    model.load_state_dict(torch.load("./Final_model.pth"))

    # Specify the folder containing the images to be classified
    image_folder = "./ResizedTestSet"

    # Predict classifications for all images in the folder
    predictions = predict_images(model, image_folder, device)

    # Save predictions to CSV
    output_file = "evaluation.csv"
    save_predictions_to_csv(predictions, output_file)

    print(f"Predictions saved to {output_file}")

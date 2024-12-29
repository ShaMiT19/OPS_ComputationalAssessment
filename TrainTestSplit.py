import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Set the paths
input_folder = "./InputFile"
train_folder = "./TrainingSet"
test_folder = "./TestSet"
input_csv = "./public.csv"
train_csv = "./train_labels.csv"
test_csv = "./test_labels.csv"

# Create output folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_csv)

# Assuming the CSV has columns 'filename' and 'label'
X = df["name"]
y = df["ground truth"]

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Function to copy files and create CSV
def process_split(file_list, labels, destination_folder, csv_path):
    data = []
    for file, label in zip(file_list, labels):
        source = os.path.join(input_folder, file)
        dest = os.path.join(destination_folder, file)
        shutil.copy2(source, dest)
        data.append({"filename": file, "label": label})

    # Create and save the CSV file
    pd.DataFrame(data).to_csv(csv_path, index=False)


# Process train and test splits
process_split(X_train, y_train, train_folder, train_csv)
process_split(X_test, y_test, test_folder, test_csv)

# Print summary
print(f"Split complete.")
print(f"Training set: {len(X_train)} images")
print(f"Test set: {len(X_test)} images")
print(f"Label distribution in training set: {y_train.value_counts(normalize=True)}")
print(f"Label distribution in test set: {y_test.value_counts(normalize=True)}")
print(f"Training CSV saved to: {train_csv}")
print(f"Test CSV saved to: {test_csv}")

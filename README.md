# README

# Trained Model weights

Link: <https://drive.google.com/drive/folders/1__KGR4xh4LRM-MexzNqTHc3s-267YUHO?usp=sharing>

# HOW TO RUN THE MODEL IN YOUR SYSTEM

1. Download the required weight files:  
    • inception_v3_google-0cc3c7bd.pth  
    • Final_model.pth  
    These can be obtained from the provided Google Drive link.
2. Compress the test dataset:  
    • Use the Compressing.py script for this purpose.  
    • Provide two inputs to the script:
    - Path to the original test dataset folder
    - Desired path for the resized images  
        • The script will generate resized images in the specified folder.
3. Environment

    • The scripts (Model.py, Test.py, and Evaluation.py) were successfully run on Kaggle using GPU T4 x2.  
    • No dependency issues were encountered.

4. Model Evaluation - (NOTE for testing you don't need to run the Model.py file. I've provided the weights file in google drive. If you wish to see whether the Model.py is working or not, then please rename the weights file from Final_model.pth to something else so that it won't cause issues with the actual Final_model.pth)

    Use the folder containing resized images for model evaluation. There are two main scripts for training and testing:
    1. Model.py:  
        • Inputs:
        - Path to the resized training images folder
        - Path to the training labels file
        - Paths to weight file (inception_v3_google-0cc3c7bd.pth)
        - Name the pth file which will save the new weights of the model. (Final_model.pth in my case)
          
    2. Test.py:  
        • Inputs:
        - Path to the resized test images folder
        - Path to the test labels file
        - Paths to both weight files (inception_v3_google-0cc3c7bd.pth and Final_model.pth)
        - Output:
            - Precision - Out of all predicted 0/1, what percentage were actually 0/1
            - Recall - Out of all actual 0/1, what percentage were correctly classified 0/1
            - F1 score - Harmonic mean of precision and recall.
            - Confusion matrix - Provides information about Correct predictions, False positives and False negatives

    3. Evaluation.py:  
        • Inputs:
        - Path to the resized test images folder
        - Paths to both weight files  
            • Output: An evaluation.csv file containing image names and corresponding model predictions  
        - Paths to both weight files (inception_v3_google-0cc3c7bd.pth and Final_model.pth)
6. I've also added the script for splitting the input dataset into training and test dataset. I takes in the path for input folder and the labels file, and splits the folder and csv into 2 parts each. With 80% on the data in training and 20% in test. Name of the file is TrainTestSplit.py
   
# Approach of the project

Since I was not able to get the access for the dataset on 26<sup>th</sup>, due to login issues with my dropbox account. I had no idea about the dataset size. I had heard that usually the dataset for medical ML models is very small. So I decided to move ahead accordingly.

Hence I decided to use transfer learning. I decided to use the InceptionV3 which is a convolutional neural network (CNN) that's used for image recognition and classification. It uses a pretrained version of the network that's been trained on over a million images from the ImageNet database.

Then I changed the final layer of the InceptionV3 CNN and added a custom layer which outputs the 2 desired classes.

This helped me sort off solve the lack of sufficient labelled dataset issue.

It was only later on 27<sup>th</sup>, that I got to know that the dataset was of 5758 images. But by then I had already added these functions in my code.

# Few key aspects of the model are as follows

## Image Preprocessing

I began by compressing the images to 299x299 dimensions, which was crucial for both storage efficiency and compatibility with the InceptionV3 model I planned to use. This compression was done using a custom script I wrote called "Compressing.py".

## Dataset Splitting

To ensure a robust evaluation, I created a script named "TrainTestSplit.py" to divide the dataset into training and testing sets. I used an 80-20 split, resulting in 4,606 images for training and 1,152 for testing. Importantly, I implemented stratified splitting to maintain class balance across both sets. I did a same split between the training and the validation set in the Model.py file. So the entire dataset is first divided into Training and test, in 80:20 split. And then the Training is further split in 80:20 split as training and validation. 

## Model Architecture

I decided to leverage transfer learning by using a pre-trained InceptionV3 model. I modified the model by replacing the final fully connected layer with an Identity function and adding a custom classifier for binary classification. This approach allowed me to benefit from the model's pre-existing knowledge while tailoring it to our specific task.

## Custom Preprocessing

Observing the distinct visual differences between sclerotic and non-sclerotic glomeruli, I developed a custom preprocessing function. This function enhances contrast and highlights structures brighter than the average background, making the key features more prominent for the model to recognize.

## Handling Class Imbalance

To address the significant class imbalance in the dataset, I implemented class weighting. This method assigns higher importance to the underrepresented class during training, helping the model learn effectively from both classes despite their uneven distribution.

## Learning Rate Scheduling

I experimented with various learning rate schedulers and found that ReduceLROnPlateau performed best for this task. This scheduler automatically adjusts the learning rate when the validation loss plateaus, leading to better convergence.

## Model Checkpointing

To ensure I retained the best-performing model, I implemented a checkpointing system that saves the model whenever the validation loss improves. This approach helps in selecting the model that generalizes best to unseen data.

# Training Pipeline

## Data Preprocessing

- Image preprocessing: The code employs a custom preprocess_image function that converts images to grayscale and applies thresholding.
- Data augmentation: Utilizes torchvision.transforms to perform random horizontal flips, tensor conversion, and normalization of the input images.

## Dataset Handling

- Custom Dataset: Implements a GlomeruliDataset class, a PyTorch Dataset subclass, for efficient loading of images and labels.
- Data Loading: Uses PyTorch's DataLoader for batching and shuffling the data during training.

## Model Architecture

- Base Model: Employs a pre-trained Inception v3 model with modifications (aux_logits disabled, fully connected layer replaced with Identity).
- Custom Classifier: Adds a sequential classifier on top of the base model, consisting of linear layers, ReLU activation, and dropout.

## Training Process

- Loss Function: Utilizes CrossEntropyLoss with class weights to handle potential class imbalance.
- Optimization: Implements Adam optimizer with an initial learning rate of 0.001.
- Learning Rate Scheduling: Employs ReduceLROnPlateau scheduler to adjust the learning rate based on validation loss.

## Evaluation

- The pipeline evaluates the model's performance using two primary metrics:
  - Validation Loss: Monitors the loss on the validation set.
  - Validation Accuracy: Calculates the accuracy of predictions on the validation set.

# Results and Analysis

The model achieved an impressive 95.75% accuracy on the test set. Here's a breakdown of the performance:

- Non-sclerotic glomeruli (Class 0):
  - Precision: 0.99
  - Recall: 0.95
  - F1-score: 0.97
- Sclerotic glomeruli (Class 1):
  - Precision: 0.83
  - Recall: 0.97
  - F1-score: 0.89

The confusion matrix revealed:

- True Negatives: 898
- False Positives: 43
- False Negatives: 6
- True Positives: 205

## Clinical Implications

The model's high sensitivity for sclerotic glomeruli (98% recall) is particularly valuable for detecting kidney damage. The perfect precision for non-sclerotic glomeruli provides high confidence in healthy assessments. While there's a slight tendency to overestimate sclerosis, this errs on the side of caution, which is often preferable in medical diagnostics.

There was another model where I had implemented a lot of the points mentioned in the Future Improvements section. The results of that are mentioned below.

The model achieved an impressive 95.23% accuracy on the test set. Here's a breakdown of the performance:

- Non-sclerotic glomeruli (Class 0):
  - Precision: 0.95
  - Recall: 0.99
  - F1-score: 0.97
- Sclerotic glomeruli (Class 1):
  - Precision: 0.94
  - Recall: 0.76
  - F1-score: 0.84

The confusion matrix revealed:

- True Negatives: 903
- False Positives: 51
- False Negatives: 11
- True Positives: 160

I decided to not go with this because it had lower recall for class 1 and also the larger number of False Negatives, which is not suitable in healthcare sector. This is mainly why I reverted back to a more simpler model which generalised better and gave good overall test accuracy.

# Future Improvements

Looking ahead, I've identified several areas for potential enhancement:

1. Data cleaning to address any errors in the dataset
2. More extensive data augmentation to increase training set diversity
3. Implementing k-fold cross-validation for more robust evaluation
4. Exploring gradient clipping to prevent exploding gradients
5. Considering the AdamW optimizer for better regularization

In conclusion, this project has yielded a highly effective model for glomeruli classification, with promising implications for assisting in kidney pathology assessments

# References
- PyTorch: <https://pytorch.org/docs/1.8.0/>
- Torchvision: <https://pytorch.org/vision/>
- Pillow (PIL): <https://pillow.readthedocs.io/en/stable/>
- Pandas: <https://pandas.pydata.org/docs/>
- NumPy: <https://numpy.org/doc/>
- Scikit-learn: <https://scikit-learn.org>
- tqdm: <https://tqdm.github.io>
- Novel Transfer Learning Approach for Medical Imaging with Limited Labeled Data: <https://pmc.ncbi.nlm.nih.gov/articles/PMC8036379/>
- Segmentation of Glomeruli Within Trichrome Images Using Deep Learning: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6612039/>
- Medical image analysis using deep learning algorithms: <https://www.frontiersin.org/journals/public-health/articles/10.3389/fpubh.2023.1273253/full>
- Machine learning in renal pathology: <https://www.frontiersin.org/journals/nephrology/articles/10.3389/fneph.2022.1007002/full>

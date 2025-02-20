#Industrial Equipment Monitoring & Machine Failure Prediction

This project aims to predict industrial equipment failures and machine failures using sensor data. The dataset contains features like temperature, pressure, humidity, and vibration for the industrial equipment monitoring dataset, and footfall, air quality, temperature, and other sensor readings for machine failure prediction.

Table of Contents

Dependencies
File Structure
How to Run the Code
Datasets and Data Preprocessing
Model Evaluation and Results

1. Dependencies
To run this project, make sure you have the following Python libraries installed:

pandas
numpy
matplotlib
seaborn
scikit-learn

How to Run the Code
To run the code, follow these steps:

Clone or download the project folder.
Install the required dependencies mentioned above.
Run the code for training and evaluation. You can execute it using the Python script directly from your command line:
bash
Copy
Edit
python train_model.py
File Details:
train_model.py: The main file where the dataset is loaded, preprocessed, and various machine learning models are trained and evaluated.
test_model.py: (Optional, if applicable) This file can contain code for testing the trained models with new or test data.
After executing the code, the output will include:

Model accuracy: Performance of different models (Decision Tree, Neural Network, AdaBoost, SVM, k-NN) on the dataset.
Classification report: Precision, recall, and F1 scores for each model.
Confusion Matrix: Graphical representation of true vs. predicted labels.


Datasets and Data Preprocessing

The datasets used in this project are:

Industrial Equipment Monitoring Dataset: Contains features like temperature, pressure, vibration, humidity, etc.
Machine Failure Prediction Dataset: Contains sensor data such as footfall, air quality, temperature, etc.
Data Preprocessing:
Missing values are handled for both numeric and categorical columns:
Numeric columns are filled with the mean value.
Categorical columns are filled with the mode (most frequent value).
Data normalization is applied to the numeric features using StandardScaler to standardize values to a common scale.
Categorical variables are encoded using One-Hot Encoding (for df1) and Label Encoding (for df2).
The data is then split into training and testing sets (80/20 split).


Model Evaluation and Results

The following machine learning models are trained and evaluated:

Decision Tree (Pruned): A decision tree classifier with pruning to avoid overfitting. Max depth is set to 5, and a minimum number of samples required to split a node is set to 10.
Neural Network (MLP Classifier): A multi-layer perceptron classifier with two hidden layers (100, 50 units), and relu activation function.
AdaBoost Classifier: An ensemble method that uses boosting to combine weak classifiers into a strong classifier.
Support Vector Machine (SVM): Uses an RBF kernel for classification. Regularization parameter C is set to 1, and gamma is set to scale.
k-Nearest Neighbors (k-NN): A non-parametric method used for classification by voting among k nearest neighbors.
Evaluation Metrics:
Accuracy Score: Percentage of correctly predicted instances.
Classification Report: Precision, recall, and F1-score for each class.
Confusion Matrix: Visualization of true vs. predicted labels.
Model Training Results
Accuracy: The accuracy for each model is displayed after training.
Learning Curves: (If applicable) You can add graphs showing training error vs. testing error across different training set sizes for each model.
Training Time: The models' runtime will be displayed, which provides insight into how efficient each algorithm is.

Conclusion
This project provides a comparison of multiple machine learning algorithms for industrial equipment monitoring and machine failure prediction. Each model's performance (accuracy, precision, recall, F1-score) will be analyzed, with suggestions for improvements based on the findings.

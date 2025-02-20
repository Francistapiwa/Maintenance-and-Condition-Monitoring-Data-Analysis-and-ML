import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
import time

# Location of data sets
file_path1 = r"C:\Users\franc\Documents\Industrial Equipment Monitoring Dataset.csv"
file_path2 = r"C:\Users\franc\Documents\Machine Failure Prediction using Sensor data.csv"

# Reading the CSV files
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)

# Get dataset information
num_instances1, num_attributes1 = df1.shape
num_instances2, num_attributes2 = df2.shape

print(f"Dataset 1: {num_instances1} instances, {num_attributes1} attributes")
print(f"Dataset 2: {num_instances2} instances, {num_attributes2} attributes")

# Conversion of column names to lower case for uniformity
df1.columns = df1.columns.str.lower().str.replace(" ", "_")
df2.columns = df2.columns.str.lower().str.replace(" ", "_")

# Standardization and replacement of missing values to prevent errors during the model training
numeric_cols1 = df1.select_dtypes(include=['float64', 'int64']).columns
categorical_cols1 = df1.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with the mean
df1[numeric_cols1] = df1[numeric_cols1].fillna(df1[numeric_cols1].mean())

# Fill missing values for categorical columns with the mode
for col in categorical_cols1:
    df1[col] = df1[col].fillna(df1[col].mode()[0])

# Repeat the process for df2
numeric_cols2 = df2.select_dtypes(include=['float64', 'int64']).columns
categorical_cols2 = df2.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with the mean
df2[numeric_cols2] = df2[numeric_cols2].fillna(df2[numeric_cols2].mean())

# Fill missing values for categorical columns with the mode (most frequent value)
for col in categorical_cols2:
    df2[col] = df2[col].fillna(df2[col].mode()[0])

# Normalization to improve system performance 
numeric_features1 = ["temperature", "pressure", "vibration", "humidity"]  # Numeric features for df1
numeric_features2 = ["footfall", "tempmode", "aq", "uss", "cs", "voc", "rp", "ip", "temperature"]  # Numeric features for df2

# Normalizing numeric columns using StandardScaler
scaler = StandardScaler()
df1[numeric_features1] = scaler.fit_transform(df1[numeric_features1])  # Normalize df1
df2[numeric_features2] = scaler.fit_transform(df2[numeric_features2])  # Normalize df2

# Encoding categorical data using One-Hot Encoding for df1 (equipment, location)
df1 = pd.get_dummies(df1, columns=["equipment", "location"], drop_first=True)

# For df2, we apply LabelEncoder to the target column (failure_type)
label_encoder = LabelEncoder()
df2["fail"] = label_encoder.fit_transform(df2["fail"])

# Here we select the features and target variables 
target_column1 = "faulty"
target_column2 = "fail"

X1 = df1.drop(target_column1, axis=1)
y1 = df1[target_column1]

X2 = df2.drop(target_column2, axis=1)
y2 = df2[target_column2]

# Split into training/testing sets (80/20 split) to ensure models are evaluated on unseen data 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_depth': [5, 10, 15, 20, None],  # Maximum depth of the tree
    'min_samples_leaf': [1, 2, 5, 10],  # Minimum samples per leaf
    'max_features': [None, 'sqrt', 'log2']  # Maximum features for the best split
}

# Initialize DecisionTreeClassifier for GridSearchCV
dt = DecisionTreeClassifier(random_state=42)

# Grid Search for Dataset 1
grid_search1 = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search1.fit(X_train1, y_train1)

# Grid Search for Dataset 2
grid_search2 = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search2.fit(X_train2, y_train2)

# Get the best estimator for each dataset
best_estimator1 = grid_search1.best_estimator_
best_estimator2 = grid_search2.best_estimator_

# Get the best parameters for both datasets
best_params1 = grid_search1.best_params_
best_params2 = grid_search2.best_params_

# Display the results
print(f"Best Estimator for Dataset 1: {best_estimator1}")
print(f"Best Estimator for Dataset 2: {best_estimator2}")

print(f"Best Parameters for Dataset 1: {best_params1}")
print(f"Best Parameters for Dataset 2: {best_params2}")

# Training and evaluating models with learning curves 
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, plot_learning_curve=False):
    """Train the model, make predictions, evaluate performance, and plot learning curves."""
    # Timing the training process
    start_time = time.time()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {model_name} Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix to visualize performance 
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot Learning Curve to show how model improves with more data 
    if plot_learning_curve:
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
        
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, label=f'Training Accuracy ({model_name})', color='blue')
        plt.plot(train_sizes, test_mean, label=f'Test Accuracy ({model_name})', color='green')
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Training with the best estimators (Grid Search results)
train_and_evaluate(best_estimator1, X_train1, X_test1, y_train1, y_test1, "Best Decision Tree (Dataset 1)", plot_learning_curve=True)
train_and_evaluate(best_estimator2, X_train2, X_test2, y_train2, y_test2, "Best Decision Tree (Dataset 2)", plot_learning_curve=True)

# Calculate balanced accuracy for both datasets
y_pred1 = best_estimator1.predict(X_test1)
balanced_accuracy1 = balanced_accuracy_score(y_test1, y_pred1)
print(f"Balanced Accuracy (Dataset 1): {balanced_accuracy1:.4f}")

y_pred2 = best_estimator2.predict(X_test2)
balanced_accuracy2 = balanced_accuracy_score(y_test2, y_pred2)
print(f"Balanced Accuracy (Dataset 2): {balanced_accuracy2:.4f}")

print("\nModel training and evaluation complete!")


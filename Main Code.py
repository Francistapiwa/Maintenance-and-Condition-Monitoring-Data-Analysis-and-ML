import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score

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

# Handling missing values
numeric_cols1 = df1.select_dtypes(include=['float64', 'int64']).columns
categorical_cols1 = df1.select_dtypes(include=['object']).columns
df1[numeric_cols1] = df1[numeric_cols1].fillna(df1[numeric_cols1].mean())
for col in categorical_cols1:
    df1[col] = df1[col].fillna(df1[col].mode()[0])

numeric_cols2 = df2.select_dtypes(include=['float64', 'int64']).columns
categorical_cols2 = df2.select_dtypes(include=['object']).columns
df2[numeric_cols2] = df2[numeric_cols2].fillna(df2[numeric_cols2].mean())
for col in categorical_cols2:
    df2[col] = df2[col].fillna(df2[col].mode()[0])

# Normalization
numeric_features1 = ["temperature", "pressure", "vibration", "humidity"]
numeric_features2 = ["footfall", "tempmode", "aq", "uss", "cs", "voc", "rp", "ip", "temperature"]

scaler = StandardScaler()
df1[numeric_features1] = scaler.fit_transform(df1[numeric_features1])
df2[numeric_features2] = scaler.fit_transform(df2[numeric_features2])

# Encoding categorical data
df1 = pd.get_dummies(df1, columns=["equipment", "location"], drop_first=True)
label_encoder = LabelEncoder()
df2["fail"] = label_encoder.fit_transform(df2["fail"])

# Feature and target selection
target_column1 = "faulty"
target_column2 = "fail"

X1, y1 = df1.drop(target_column1, axis=1), df1[target_column1]
X2, y2 = df2.drop(target_column2, axis=1), df2[target_column2]

# Train-test split (80/20)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.histplot(y1, kde=False, bins=10, color='skyblue', label='Faulty Class Distribution')
plt.title('Class Distribution - Dataset 1 (Faulty)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1])
plt.legend()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(y2, kde=False, bins=10, color='salmon', label='Failure Type Class Distribution')
plt.title('Class Distribution - Dataset 2 (Failure Type)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2])
plt.legend()
plt.show()

# Training and evaluation function
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, plot_learning_curve=False):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    training_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {model_name} Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.4f} seconds")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Learning Curve
    if plot_learning_curve:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=cv, scoring="accuracy", 
                                                                train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
        train_mean, test_mean = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_mean, label=f'Training Accuracy ({model_name})', color='blue')
        plt.plot(train_sizes, test_mean, label=f'Test Accuracy ({model_name})', color='green')
        plt.xlabel('Training Size')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"learning_curve_{model_name}.png")
        plt.show()

# Train models
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
train_and_evaluate(dt, X_train1, X_test1, y_train1, y_test1, "Decision Tree (Pruned)", plot_learning_curve=True)

nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, activation='relu', solver='adam', random_state=42)
train_and_evaluate(nn, X_train1, X_test1, y_train1, y_test1, "Neural Network", plot_learning_curve=True)

boost = AdaBoostClassifier(n_estimators=50, random_state=42)
train_and_evaluate(boost, X_train1, X_test1, y_train1, y_test1, "Boosting (AdaBoost)", plot_learning_curve=True)

svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
train_and_evaluate(svm_rbf, X_train1, X_test1, y_train1, y_test1, "SVM (RBF Kernel)", plot_learning_curve=True)

svm_linear = SVC(kernel='linear', C=1, random_state=42)
train_and_evaluate(svm_linear, X_train1, X_test1, y_train1, y_test1, "SVM (Linear Kernel)", plot_learning_curve=True)

knn = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate(knn, X_train1, X_test1, y_train1, y_test1, "k-Nearest Neighbors (k-NN)", plot_learning_curve=True)

print("\nModel training and evaluation complete!")

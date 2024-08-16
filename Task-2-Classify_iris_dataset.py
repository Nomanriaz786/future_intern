import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert the data into a Pandas DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Print the feature names and target names
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Print the first few samples in the dataset
print("First 5 samples:")
for i in range(5):
    print(f"Sample {i+1}: {data.iloc[i].values} (Class: {data['target'].iloc[i]}, Species: {iris.target_names[data['target'].iloc[i]]})")

# Visualize the data using a scatter plot
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()

# Split the data into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to train and evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model.__class__.__name__} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model.__class__.__name__} Precision:", precision_score(y_test, y_pred, average='weighted'))
    print(f"{model.__class__.__name__} Recall:", recall_score(y_test, y_pred, average='weighted'))
    print(f"{model.__class__.__name__} F1-score:", f1_score(y_test, y_pred, average='weighted'))

# Logistic Regression
evaluate_model(LogisticRegression(), X_train, y_train, X_test, y_test)

# Decision Tree
evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test)

# Support Vector Machine (SVM)
evaluate_model(SVC(), X_train, y_train, X_test, y_test)
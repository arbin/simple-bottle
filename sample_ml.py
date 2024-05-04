import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Display the first few rows of the dataset
# print(df.head())


# Split the dataset into features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']
#
# # Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#
#
# # Create and train the K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
#
# Predict classes for test set
y_pred = knn.predict(X_test_scaled)
#
# # Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#
# # Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


"""
Explanation of Steps:
Step 1: We import necessary libraries including numpy, pandas, and scikit-learn components for dataset loading, model training, and evaluation.
Step 2: We load the Iris dataset using load_iris() and convert it into a pandas DataFrame. The dataset contains features like sepal length, sepal width, petal length, and petal width along with corresponding target labels representing different iris species.
Step 3: We split the data into training and testing sets using train_test_split(), and then standardize the feature data using StandardScaler() to ensure all features are on a similar scale.
Step 4: We initialize a K-Nearest Neighbors (KNN) classifier (KNeighborsClassifier) and train it on the scaled training data (X_train_scaled, y_train).
Step 5: We use the trained model to make predictions on the test data (X_test_scaled) and evaluate its performance using metrics like accuracy and classification report (accuracy_score, classification_report).
T
his example showcases a typical workflow of training and evaluating a machine learning model using scikit-learn in Python. 
You can further explore different algorithms, 
tune hyperparameters, and 
perform more sophisticated tasks based on this foundation.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Data split parameters
SEED_STATE = 23
TEST_SIZE = 0.2


# Task 1 – Common dataset

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame with features
df = pd.DataFrame(data.data, columns=data.feature_names)

# Separate features and labels
X = df
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED_STATE, stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


# Task 2 – Model scikit-learn

# Create a pipeline with scaling and Random Forest
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=SEED_STATE))
])

# Train the model
model.fit(X_train, y_train)

# Predict labels for the test set
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
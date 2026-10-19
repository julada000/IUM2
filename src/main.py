import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold, cross_val_score

import time
from pathlib import Path

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


# Task 3 - PyTorch model

# Set random seed
torch.manual_seed(SEED_STATE)

# Standardize the data
scaler_torch = StandardScaler()

X_train_scaled = scaler_torch.fit_transform(X_train)
X_test_scaled = scaler_torch.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Initialize the model
model_torch = NeuralNetwork(
    input_size=X_train.shape[1],
    num_classes=len(np.unique(y_train))
)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.001)

# Set number of epochs
epochs = 100

# Train the model
model_torch.train()

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model_torch(X_train_tensor)

    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    # Display training progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model_torch.eval()

with torch.no_grad():
    outputs = model_torch(X_test_tensor)
    y_pred_torch = torch.argmax(outputs, dim=1).numpy()

# Calculate model accuracy
accuracy_torch = accuracy_score(y_test, y_pred_torch)

print("Accuracy PyTorch:", accuracy_torch)


# Task 4 – Cross-validation

# Define stratified 5-fold cross-validation
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED_STATE
)

start_sklearn = time.perf_counter()

# Evaluate the scikit-learn model
scores_sklearn = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring="accuracy"
)

time_sklearn = time.perf_counter() - start_sklearn

# Calculate mean accuracy and standard deviation
mean_sklearn = np.mean(scores_sklearn)
std_sklearn = np.std(scores_sklearn)

# Calculate additional cross-validation metrics
precision_sklearn = cross_val_score(
    model, X_train, y_train, cv=skf, scoring="precision"
)

recall_sklearn = cross_val_score(
    model, X_train, y_train, cv=skf, scoring="recall"
)

f1_sklearn = cross_val_score(
    model, X_train, y_train, cv=skf, scoring="f1"
)


print("Accuracy for each fold:", scores_sklearn)
print(f"Mean accuracy: {mean_sklearn:.4f}")
print(f"Standard deviation: {std_sklearn:.4f}")
print(f"Mean precision: {np.mean(precision_sklearn):.4f}")
print(f"Mean recall: {np.mean(recall_sklearn):.4f}")
print(f"Mean F1-score: {np.mean(f1_sklearn):.4f}")


# Task 5 – PyTorch cross-validation

start_torch = time.perf_counter()

# Store accuracy scores for each fold
scores_torch = []
precision_torch = []
recall_torch = []
f1_torch = []

# Perform stratified 5-fold cross-validation
for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_train, y_train), start=1
):
    print(f"\nFold {fold}/5")

    # Split data into training and validation sets
    X_fold_train = X_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]

    # Scale data for each fold
    scaler_fold = StandardScaler()

    X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
    X_fold_val_scaled = scaler_fold.transform(X_fold_val)

    # Convert data to PyTorch tensors
    X_fold_train_tensor = torch.tensor(
        X_fold_train_scaled, dtype=torch.float32
    )
    X_fold_val_tensor = torch.tensor(
        X_fold_val_scaled, dtype=torch.float32
    )

    y_fold_train_tensor = torch.tensor(
        y_fold_train, dtype=torch.long
    )
    y_fold_val_tensor = torch.tensor(
        y_fold_val, dtype=torch.long
    )

    # Initialize a new model for each fold
    torch.manual_seed(SEED_STATE + fold)

    model_fold = NeuralNetwork(
        input_size=X_train.shape[1],
        num_classes=len(np.unique(y_train))
    )

    # Define loss function and optimizer
    criterion_fold = nn.CrossEntropyLoss()

    optimizer_fold = optim.Adam(
        model_fold.parameters(),
        lr=0.001
    )

    # Train the model
    model_fold.train()

    for epoch in range(100):
        optimizer_fold.zero_grad()

        outputs = model_fold(X_fold_train_tensor)

        loss = criterion_fold(outputs, y_fold_train_tensor)

        loss.backward()
        optimizer_fold.step()

    # Evaluate the model on validation data
    model_fold.eval()

    with torch.no_grad():
        outputs = model_fold(X_fold_val_tensor)
        predictions = torch.argmax(outputs, dim=1)

        accuracy = (
            (predictions == y_fold_val_tensor)
            .float()
            .mean()
            .item()
        )

    # Store and display fold accuracy
    scores_torch.append(accuracy)
    
    # Calculate additional metrics for the current fold
    y_true_fold = y_fold_val_tensor.numpy()
    y_pred_fold = predictions.numpy()

    precision_torch.append(
        precision_score(y_true_fold, y_pred_fold, zero_division=0)
    )

    recall_torch.append(
        recall_score(y_true_fold, y_pred_fold, zero_division=0)
    )

    f1_torch.append(
        f1_score(y_true_fold, y_pred_fold, zero_division=0)
    )


    print(f"Accuracy fold {fold}: {accuracy:.4f}")

time_torch = time.perf_counter() - start_torch

# Calculate mean accuracy and standard deviation
mean_torch = np.mean(scores_torch)
std_torch = np.std(scores_torch)

print("Accuracy for each fold:", scores_torch)
print(f"Mean accuracy PyTorch: {mean_torch:.4f}")
print(f"Standard deviation PyTorch: {std_torch:.4f}")

# Calculate mean and standard deviation for additional metrics
mean_precision_torch = np.mean(precision_torch)
mean_recall_torch = np.mean(recall_torch)
mean_f1_torch = np.mean(f1_torch)

print(f"Mean precision PyTorch: {mean_precision_torch:.4f}")
print(f"Mean recall PyTorch: {mean_recall_torch:.4f}")
print(f"Mean F1-score PyTorch: {mean_f1_torch:.4f}")


# Task 6 – Model comparison

# Summarize cross-validation results
results = pd.DataFrame({
    "Model": ["scikit-learn", "PyTorch"],
    "Średnia dokładność": [mean_sklearn, mean_torch],
    "Odch. standardowe": [std_sklearn, std_torch],
    "Precyzja": [np.mean(precision_sklearn), np.mean(precision_torch)],
    "Czułość": [np.mean(recall_sklearn), np.mean(recall_torch)],
    "F1-score": [np.mean(f1_sklearn), np.mean(f1_torch)],
    "Czas walidacji [s]": [time_sklearn, time_torch]
})

# Compare models based on mean accuracy
if mean_sklearn > mean_torch:
    conclusion = "Model scikit-learn uzyskał wyższą średnią dokładność."
elif mean_torch > mean_sklearn:
    conclusion = "Model PyTorch uzyskał wyższą średnią dokładność."
else:
    conclusion = "Oba modele uzyskały taką samą średnią dokładność."

# Prepare and save the report as a text file
report = f"""PORÓWNANIE MODELI SCIKIT-LEARN I PYTORCH

Wyniki 5-krotnej walidacji krzyżowej:
{results.to_string(index=False, float_format=lambda x: f'{x:.4f}')}

Dokładność w poszczególnych foldach:
scikit-learn: {np.round(scores_sklearn, 4)}
PyTorch:      {np.round(scores_torch, 4)}

Porównanie złożoności kodu:
scikit-learn: mniej kodu dzięki Pipeline i cross_val_score.
PyTorch: więcej kodu ze względu na definicję sieci i ręczną pętlę treningową.

Podsumowanie:
{conclusion}
Czas walidacji scikit-learn: {time_sklearn:.2f} s.
Czas walidacji PyTorch: {time_torch:.2f} s.
Podane czasy obejmują walidację krzyżową, a nie wyłącznie trening modeli.
"""

Path("wyniki.txt").write_text(report, encoding="utf-8")
print(report)
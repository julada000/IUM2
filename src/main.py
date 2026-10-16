import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim



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

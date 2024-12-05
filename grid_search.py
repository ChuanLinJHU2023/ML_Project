import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Step 1: Load the datasets
train_df = pd.read_csv('train.csv')
validation_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test.csv')

# Step 2: Prepare the features and target variable
X_train = train_df.drop(columns=['Stage']).values
y_train = train_df['Stage'].values
X_val = validation_df.drop(columns=['Stage']).values
y_val = validation_df['Stage'].values
X_test = test_df.drop(columns=['Stage']).values
y_test = test_df['Stage'].values

# Step 3: Standardize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 4: Prepare data for PyTorch
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val_scaled)
y_val_tensor = torch.LongTensor(y_val)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Step 5: Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 6: Create a simple MLP model
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Apply ReLU except for the last layer
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Step 7: Grid Search Setup
hidden_units_options = [[64], [128], [64, 64], [128, 64]]  # Different architectures
learning_rates = [0.001, 0.01, 0.1]  # Different learning rates
results = []

num_epochs = 500
patience = 50  # For early stopping

# Grid Search Loop
for hidden_units in hidden_units_options:
    for lr in learning_rates:
        model = MLP([X_train_tensor.shape[1]] + hidden_units + [4]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor.to(device))
            loss = criterion(outputs, y_train_tensor.to(device))
            loss.backward()
            optimizer.step()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter if we improved
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            _, test_predictions = torch.max(test_outputs, 1)
            test_accuracy = accuracy_score(y_test, test_predictions.cpu())

        # Save results
        results.append({
            'hidden_units': str(hidden_units),
            'learning_rate': lr,
            'test_accuracy': test_accuracy,
        })

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('grid_search.csv', index=False)

print("Grid search completed. Results saved to grid_search.csv.")
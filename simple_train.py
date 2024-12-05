import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 7: Initialize the model
input_size = X_train_tensor.shape[1]
hidden_size = 100  # You can adjust this
output_size = 4  # For classes 0, 1, 2, 3

model = MLP(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 8: Training with early stopping
num_epochs = 500
patience = 50  # Number of epochs to wait for improvement
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))

    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()

    # Track training loss
    train_losses.append(loss.item())

    # Calculate training accuracy
    _, train_predictions = torch.max(outputs, 1)
    train_accuracy = (train_predictions.cpu() == y_train_tensor).float().mean().item()
    train_accuracies.append(train_accuracy)

    # Validation loss and accuracy
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor.to(device))
        val_loss = criterion(val_outputs, y_val_tensor.to(device))
        val_losses.append(val_loss.item())

        _, val_predictions = torch.max(val_outputs, 1)
        val_accuracy = (val_predictions.cpu() == y_val_tensor).float().mean().item()
        val_accuracies.append(val_accuracy)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter if we improved
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, '
              f'Validation Loss: {val_loss.item():.4f}, Accuracy: {val_accuracy:.4f}')

# Step 9: Plot Losses and Accuracies
plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 for better visibility
plt.legend()

plt.tight_layout()
plt.show()

# Step 10: Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor.to(device))
    _, test_predictions = torch.max(test_outputs, 1)

# Step 11: Print evaluation results
print("Test Set Performance:")
print("Accuracy:", accuracy_score(y_test, test_predictions.cpu()))
print(classification_report(y_test, test_predictions.cpu()))
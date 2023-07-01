import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Read the data using pandas
df = pd.read_csv("C:/Users/Owner/Desktop/cs stuff/Open Source/nn-nhl/scripts/ml_X.csv")
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).cuda()
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).cuda()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).cuda()
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).cuda()

# Create a DataLoader for efficient batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the model architecture
class ANN_Model(nn.Module):
    def __init__(self, input_size):
        super(ANN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Create an instance of the model
model = ANN_Model(input_size=X_train.shape[1]).cuda()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
# model.to(device)

# Define the loss function and optimizer
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        outputs = model(batch_inputs)
        loss = loss_function(outputs, batch_targets.unsqueeze(1))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print epoch loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluation
model.eval()
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
outputs = model(X_test_tensor)
predicted_labels = torch.round(torch.sigmoid(outputs))
accuracy = (predicted_labels == y_test_tensor.unsqueeze(1)).sum().item() / len(y_test_tensor)
print(f"Accuracy: {accuracy}")

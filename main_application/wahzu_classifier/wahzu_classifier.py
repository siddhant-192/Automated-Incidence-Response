import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the scaler used during training
scaler = StandardScaler()
# Ensure to fit this scaler on your original training data during initialization

# Define the model architecture (same as during training)
class FinalBestModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super(FinalBestModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# Initialize the model with training architecture parameters
input_dim =  # Your input dimension here (e.g., number of features after one-hot encoding)
output_dim =  # Number of output classes
hidden_dim1 = 192
hidden_dim2 = 32
dropout_rate = 0.3205471047392418

model = FinalBestModel(input_dim, output_dim, hidden_dim1, hidden_dim2, dropout_rate)

# Load the saved model weights
model.load_state_dict(torch.load("final_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the classifier function
def wahzu_classifier(wahzu_logs):
    # Convert logs to DataFrame if they aren’t already
    if isinstance(wahzu_logs, dict):  # Assuming logs are in dictionary format
        wahzu_logs = pd.DataFrame([wahzu_logs])
    elif isinstance(wahzu_logs, list):  # Assuming list of dictionaries
        wahzu_logs = pd.DataFrame(wahzu_logs)

    # Preprocess: one-hot encode and align columns
    wahzu_logs = pd.get_dummies(wahzu_logs)
    wahzu_logs = wahzu_logs.reindex(columns=X.columns, fill_value=0)

    # Standardize using the original scaler
    wahzu_logs = scaler.transform(wahzu_logs)

    # Convert to PyTorch tensor
    wahzu_logs_tensor = torch.tensor(wahzu_logs, dtype=torch.float32)

    # Run the model for classification
    with torch.no_grad():
        outputs = model(wahzu_logs_tensor)
        _, predicted_classes = torch.max(outputs, 1)

    # Filter logs classified as flagged (1) and store them in a list
    flagged_logs = wahzu_logs[predicted_classes.numpy() == 1].to_dict(orient='records')

    return flagged_logs
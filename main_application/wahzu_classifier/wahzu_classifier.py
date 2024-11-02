import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import ast
import os

# Get the directory of this file (classifier.py)
base_dir = os.path.dirname(__file__)

# Load the pre-fitted scaler and encoder
scaler = joblib.load("wahzu_classifier/wahzu_scaler.pkl")  # Assuming you've saved the scaler
one_hot_encoder = joblib.load("wahzu_classifier/wahzu_one_hot_encoder.pkl")  # Load the one-hot encoder
model_path =  "wahzu_classifier/wahzu_model.pth"

# Define neural network model architecture to match training configuration
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

# Load the model weights and set to evaluation mode
model = FinalBestModel(input_dim=192, output_dim=2, hidden_dim1=192, hidden_dim2=32, dropout_rate=0.32)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the classifier function
def wahzu_classifier(wahzu_logs):
    # Convert logs to DataFrame
    classify_df = pd.DataFrame(wahzu_logs)

    # Replace empty strings with NaN
    classify_df.replace(' ', np.nan, inplace=True)
    classify_df.columns = classify_df.columns.str.replace('^_source.', '', regex=True)

    # Define columns as per training configuration
    final_cols = [
        'agent.name', 'data.win.system.eventID', 'data.win.system.channel',
        'data.win.system.severityValue', 'data.win.system.providerName', 'rule.firedtimes',
        'rule.level', 'rule.groups', 'location', 'data.win.eventdata.logonProcessName',
        'data.win.eventdata.elevatedToken', 'data.win.eventdata.processName',
        'data.win.eventdata.targetDomainName', 'data.win.eventdata.logonType',
        'rule.mitre.technique', 'rule.mitre.tactic', 'syscheck.path', 'syscheck.event',
        'syscheck.value_name', 'syscheck.win_perm_after', 'data.win.eventdata.p1',
        'data.win.eventdata.serviceType', 'data.vulnerability.severity', 'data.vulnerability.cve',
        'data.vulnerability.cvss.cvss3.base_score', 'data.win.eventdata.originalFileName',
        'data.win.eventdata.image'
    ]

    classify_df = classify_df[final_cols]

    # Fill NaN and encode categorical columns using the pre-fitted one-hot encoder
    one_hot_columns = ['agent.name', 'data.win.system.eventID', 'data.win.system.channel', 'location',
                       'data.win.system.providerName', 'rule.mitre.technique', 'rule.mitre.tactic', 'syscheck.event']
    for column in one_hot_columns:
        classify_df[column] = classify_df[column].fillna('unknown').astype(str)
    
    # Apply the loaded one-hot encoder
    encoded = one_hot_encoder.transform(classify_df[one_hot_columns])
    encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))
    classify_df = pd.concat([classify_df.drop(columns=one_hot_columns), encoded_df], axis=1)

    # Log transform numerical columns and handle any non-numeric or NaN values
    log_transform_columns = ['rule.firedtimes']
    for column in log_transform_columns:
        classify_df[column] = pd.to_numeric(classify_df[column], errors='coerce').apply(lambda x: np.log(x) if x > 0 else 0)
    
    # Handle categorical mappings for label encoding
    label_encoding_column_order_mapping = {
        'data.win.system.severityValue': [np.nan, 'INFORMATION', 'AUDIT_SUCCESS', 'WARNING', 'ERROR', 'AUDIT_FAILURE'],
    }
    for column, order in label_encoding_column_order_mapping.items():
        order_mapping = {value: idx for idx, value in enumerate(order)}
        classify_df[column] = classify_df[column].map(order_mapping)

    # Align columns with the training data columns
    classify_df = classify_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Ensure all columns are numeric
    for column in classify_df.columns:
        classify_df[column] = pd.to_numeric(classify_df[column], errors='coerce').fillna(0)

    # Scale the data for model compatibility
    numeric_df = scaler.transform(classify_df)
    
    # Convert to tensor for model
    logs_tensor = torch.tensor(numeric_df, dtype=torch.float32)

    # Predict using the model
    with torch.no_grad():
        outputs = model(logs_tensor)
        _, predicted_classes = torch.max(outputs, 1)
    
    # Collect flagged logs
    flagged_logs = classify_df[predicted_classes.numpy() == 1].to_dict(orient='records')

    print("Total: ", len(wahzu_logs))
    print("Flagged: ", len(flagged_logs))

    return flagged_logs

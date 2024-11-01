import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import ast

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################



safe_df = pd.read_csv('safe.csv')
threat_df = pd.read_csv('threat.csv')

safe_df.replace(' ', np.nan, inplace=True)

threat_df.replace(' ', np.nan, inplace=True)

extra_columns = [
    "_source.data.win.eventdata.originalFileName",
    "_source.data.win.eventdata.image",
    "_source.data.win.eventdata.description",
    "_source.data.win.eventdata.parentCommandLine",
    "_source.data.win.eventdata.parentImage",
    "_source.data.win.eventdata.commandLine",
    "_source.data.win.eventdata.integrityLevel",
    "_source.data.win.eventdata.user",
    "_source.data.win.eventdata.targetFilename",
    "_source.data.win.eventdata.appName",
    "_source.data.win.eventdata.moduleName",
    "_source.data.win.eventdata.appPath"
]

safe_df[extra_columns] = np.nan

final_cols = [
    'agent.name',
    'data.win.system.eventID',
    'data.win.system.channel',
    'data.win.system.severityValue',
    'data.win.system.providerName',
    'rule.firedtimes',#
    'rule.level',
    'rule.groups',
    'location',
    'data.win.eventdata.logonProcessName',
    'data.win.eventdata.elevatedToken',
    'data.win.eventdata.processName',#
    'data.win.eventdata.targetDomainName',
    'data.win.eventdata.logonType',#
    'rule.mitre.technique',
    'rule.mitre.tactic',
    'syscheck.path',
    'syscheck.event',
    'syscheck.value_name',
    'syscheck.win_perm_after',
    'data.win.eventdata.p1',
    'data.win.eventdata.serviceType',#
    'data.vulnerability.severity',#
    'data.vulnerability.cve',#
    'data.vulnerability.cvss.cvss3.base_score',# scale
    'data.win.eventdata.originalFileName',
    'data.win.eventdata.image',
    #'data.win.eventdata.integrityLevel',
    'Flag'
]

print(len(final_cols))

safe_df.columns = safe_df.columns.str.replace('^_source.', '', regex=True)
threat_df.columns = threat_df.columns.str.replace('^_source.', '', regex=True)

safe_final_df = safe_df[final_cols]
threat_final_df = threat_df[final_cols]



############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################

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


# Define the classifier function
def wahzu_classifier(wahzu_logs):
    n_logs = len(wahzu_logs)

    # Convert logs to DataFrame if they aren’t already
    classify_df = pd.DataFrame(wahzu_logs)

    classify_df.replace(' ', np.nan, inplace=True)

    # Assuming data is comming in ways similar to threat logs
    # extra_columns = [
    #     "_source.data.win.eventdata.originalFileName",
    #     "_source.data.win.eventdata.image",
    #     "_source.data.win.eventdata.description",
    #     "_source.data.win.eventdata.parentCommandLine",
    #     "_source.data.win.eventdata.parentImage",
    #     "_source.data.win.eventdata.commandLine",
    #     "_source.data.win.eventdata.integrityLevel",
    #     "_source.data.win.eventdata.user",
    #     "_source.data.win.eventdata.targetFilename",
    #     "_source.data.win.eventdata.appName",
    #     "_source.data.win.eventdata.moduleName",
    #     "_source.data.win.eventdata.appPath"
    # ]

    # classify_df[extra_columns] = np.nan

    final_cols = [
        'agent.name',
        'data.win.system.eventID',
        'data.win.system.channel',
        'data.win.system.severityValue',
        'data.win.system.providerName',
        'rule.firedtimes',#
        'rule.level',
        'rule.groups',
        'location',
        'data.win.eventdata.logonProcessName',
        'data.win.eventdata.elevatedToken',
        'data.win.eventdata.processName',#
        'data.win.eventdata.targetDomainName',
        'data.win.eventdata.logonType',#
        'rule.mitre.technique',
        'rule.mitre.tactic',
        'syscheck.path',
        'syscheck.event',
        'syscheck.value_name',
        'syscheck.win_perm_after',
        'data.win.eventdata.p1',
        'data.win.eventdata.serviceType',#
        'data.vulnerability.severity',#
        'data.vulnerability.cve',#
        'data.vulnerability.cvss.cvss3.base_score',# scale
        'data.win.eventdata.originalFileName',
        'data.win.eventdata.image',
        #'data.win.eventdata.integrityLevel',
        'Flag'
    ]

    classify_df.columns = threat_df.columns.str.replace('^_source.', '', regex=True)

    classify_final_df = threat_df[final_cols]

    combined_df = pd.concat([safe_final_df, threat_final_df, classify_final_df], axis=0)

    filtering_df = combined_df

    flagged_logs = []

    # Flag rows where 'data.win.eventdata.serviceType' is 'kernel mode driver'
    flagged_rows = combined_df[combined_df['data.win.eventdata.serviceType'] == 'kernel mode driver']

    # Add these flagged rows to the flagged_logs list
    new_flagged_logs = flagged_rows.to_dict('records')  # Convert the rows to dictionary format for easier logging

    flagged_logs = flagged_logs + new_flagged_logs

    # Remove the flagged rows from the original dataframe
    combined_df = combined_df[combined_df['data.win.eventdata.serviceType'] != 'kernel mode driver']

    # Flag rows where 'data.win.eventdata.serviceType' is 'kernel mode driver'
    flagged_rows = combined_df[combined_df['data.vulnerability.severity'] == 'Critical']

    # Add these flagged rows to the flagged_logs list
    new_flagged_logs = flagged_rows.to_dict('records')  # Convert the rows to dictionary format for easier logging

    flagged_logs = flagged_logs + new_flagged_logs

    # Remove the flagged rows from the original dataframe
    combined_df = combined_df[combined_df['data.vulnerability.severity'] != 'Critical']

    combined_df['data.vulnerability.cve'] = combined_df['data.vulnerability.cve'].astype(str) 

    # Flag rows where 'data.win.eventdata.serviceType' is 'kernel mode driver'
    flagged_rows = combined_df[combined_df['data.vulnerability.cve'] != 'nan']

    # Add these flagged rows to the flagged_logs list
    new_flagged_logs = flagged_rows.to_dict('records')  # Convert the rows to dictionary format for easier logging

    flagged_logs = flagged_logs + new_flagged_logs

    # Remove the flagged rows from the original dataframe
    combined_df = combined_df[combined_df['data.vulnerability.cve'] == 'nan']


    # Replace NaN values with 'unmodified' in the 'syscheck.event' column
    combined_df['syscheck.event'] = combined_df['syscheck.event'].fillna('unmodified')

    combined_df['data.win.eventdata.image'] = combined_df['data.win.eventdata.image'].apply(lambda x: 1 if isinstance(x, str) and x.startswith('C:\\Windows\\System32\\') else 0)

    # Columns to One-Hot Encode
    one_hot_columns = ['agent.name', 'data.win.system.eventID', 'data.win.system.channel', 'location', 'data.win.system.providerName', 'rule.mitre.technique', 'rule.mitre.tactic', 'syscheck.event']

    # Columns to Log Transform
    log_transform_columns = ['rule.firedtimes']

    # Columns to Label Encode
    # Structured dictionary to store column names and their respective orderings
    label_encoding_column_order_mapping = {
        'data.win.system.severityValue': [np.nan, 'INFORMATION', 'AUDIT_SUCCESS', 'WARNING', 'ERROR', 'AUDIT_FAILURE'],
    }

    # Dictionary with acceptable values for specific columns
    acceptable_values = {
        'data.win.eventdata.targetDomainName': ['NT AUTHORITY'],  # Only 'NT AUTHORITY' is acceptable for column 'data.win.eventdata.targetDomainName'
        'data.win.eventdata.originalFileName': ['wannacry.exe', 'notpetya.exe', 'trickbot.exe', 'emotet.exe', 'ryuk.exe', 'locky.exe', 'cryptolocker.exe', 'keylogger.exe', 'winspy.exe', 'darkcomet.exe', 'nanocore.exe', 'teamviewer.exe', 'anydesk.exe', 'radmin.exe', 'vncserver.exe', 'remcmd.exe', 'mimikatz.exe', 'procdump.exe', 'dumpert.exe', 'pwdump.exe', 'nmap.exe', 'angryipscanner.exe', 'metasploit.exe', 'sqlmap.exe', 'xmrig.exe', 'minerd.exe', 'cryptonight.exe', 'ccminer.exe', 'svchost.exe', 'explorer.exe', 'lsass.exe', 'csrss.exe', 'winlogon.exe', 'wscript.exe', 'cscript.exe', 'powershell.exe', 'mshta.exe', 'regsvr32.exe', 'installutil.exe', 'msiexec.exe', 'schtasks.exe', 'certutil.exe', 'bitsadmin.exe', 'ftp.exe', 'sc.exe', 'driverquery.exe', 'rundll32.exe', 'taskeng.exe', 'conhost.exe'] ,
        'data.win.eventdata.logonType': ['wannacry.exe', 'notpetya.exe'], # New list required
        'data.win.eventdata.processName': ['wannacry.exe', 'notpetya.exe'] # New list required
        
    }

    # List of columns for missing value indicator
    missing_value_columns = ['data.win.eventdata.logonProcessName ', 'data.win.eventdata.elevatedToken', 'syscheck.path', 'syscheck.value_name', 'syscheck.win_perm_after', 'data.win.eventdata.p1']

    # List of columns for scaling
    scaling_columns = []

    for column in one_hot_columns:
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid dummy variable trap
        encoded = encoder.fit_transform(combined_df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))

        # Reset the index to avoid concatenation issues
        combined_df = combined_df.drop(column, axis=1).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)

        # Concatenate the two DataFrames
        combined_df = pd.concat([combined_df, encoded_df], axis=1)

    # Log Transformation
    for column in log_transform_columns:
        # Check for non-positive values to avoid issues with log(0) or log of negative numbers
        if (combined_df[column] <= 0).any():
            raise ValueError(f"Column '{column}' contains non-positive values, cannot apply log transform.")
        combined_df[column] = np.log(combined_df[column])

    # Function to map each column based on its order
    def label_encode_with_custom_order(df, label_encoding_column_order_mapping):
        for column, order in label_encoding_column_order_mapping.items():
            # Create a dictionary mapping the custom order to numerical values
            order_mapping = {value: idx for idx, value in enumerate(order)}
            
            # Apply the mapping to the column
            df[column] = df[column].map(order_mapping)
        
        return df

    # Apply the custom encoding
    combined_df = label_encode_with_custom_order(combined_df, label_encoding_column_order_mapping)

    # Acceptable Values Mapping
    for column, acceptable in acceptable_values.items():
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].apply(lambda x: 1 if x in acceptable else 0)

    # Missing Value Indicator
    for column in missing_value_columns:
        if column in combined_df.columns:
            combined_df[column] = combined_df[column].apply(lambda x: 0 if pd.isnull(x) else 1)

    # Scale data.vulnerability.cvss.cvss3.base_score to get between 0-1
    combined_df['data.vulnerability.cvss.cvss3.base_score'] = combined_df['data.vulnerability.cvss.cvss3.base_score'].fillna(0)  # Replace NaN with 0
    combined_df['data.vulnerability.cvss.cvss3.base_score'] = combined_df['data.vulnerability.cvss.cvss3.base_score'].astype(float)  # Convert to float for division
    combined_df['data.vulnerability.cvss.cvss3.base_score'] = combined_df['data.vulnerability.cvss.cvss3.base_score'].apply(lambda x: x / 10)  # Scale the values



    # Convert 'rule.level' to numeric, forcing invalid parsing to NaN
    combined_df['rule.level'] = pd.to_numeric(combined_df['rule.level'], errors='coerce')

    # Define the bins and labels
    bins = [0, 4, 10, 15]
    labels = ['LOW', 'MED', 'HIGH']

    # Binning the 'rule.level' column
    combined_df['rule.level'] = pd.cut(
        combined_df['rule.level'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True
    )

    # Handle any NaN values that may result from out-of-bounds values
    combined_df['rule.level'] = combined_df['rule.level'].fillna('LOW')

    # Map the binned categories to specific numerical values
    category_mapping = {'LOW': 0, 'MED': 1, 'HIGH': 2}
    combined_df['rule.level'] = combined_df['rule.level'].map(category_mapping)



    # Step 1: Parse the 'rule.groups' column from string representation of lists to actual lists
    combined_df['rule.groups'] = combined_df['rule.groups'].apply(ast.literal_eval)

    # Step 2: Extract all unique individual values from the 'rule.groups' column
    unique_values = set()
    for groups in combined_df['rule.groups']:
        unique_values.update(groups)

    unique_values = sorted(unique_values)  # Optional: sort the unique values for consistency

    # Step 3: Create a new column for each unique value and encode it
    for value in unique_values:
        combined_df[value] = combined_df['rule.groups'].apply(lambda x: 1 if value in x else 0)

    # Step 4: Remove the original 'rule.groups' column
    combined_df = combined_df.drop('rule.groups', axis=1)

    logs_to_classify = combined_df.tail(n_logs)

    # Initialize the model with training architecture parameters
    input_dim =  logs_to_classify.shape[1]
    output_dim =  2
    hidden_dim1 = 192
    hidden_dim2 = 32
    dropout_rate = 0.3205471047392418

    model = FinalBestModel(input_dim, output_dim, hidden_dim1, hidden_dim2, dropout_rate)

    # Load the saved model weights
    model.load_state_dict(torch.load("final_model.pth"))
    model.eval()  # Set the model to evaluation mode

    

    # Preprocess: one-hot encode and align columns
    wahzu_logs = pd.get_dummies(logs_to_classify)
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
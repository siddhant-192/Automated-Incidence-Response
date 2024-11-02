import pandas as pd
from wahzu_classifier import wahzu_classifier  # Replace 'your_module' with the actual module name if the classifier code is saved in a separate file

# Load the combined log data
combined_df = pd.read_csv('combined.csv')

# Load wahzu logs and call wahzu_classifier
wahzu_logs = pd.read_csv('combined.csv').to_dict(orient='records')
flagged_logs = wahzu_classifier(wahzu_logs)

# Print flagged logs
print("Flagged Logs:")
for log in flagged_logs:
    print(log)

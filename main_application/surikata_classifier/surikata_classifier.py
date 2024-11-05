import pandas as pd

def surikata_classifier(surikata_logs):
    surikata_df = surikata_logs
    flagged_logs = []
    
    # Iterate over each row in the DataFrame
    for _, row in surikata_df.iterrows():
        # Check if "ssh" is in any of the values in the row
        if any("ssh" in str(value) for value in row):
            flagged_logs.append(row.to_dict())  # Convert row to a dictionary and add to flagged list
    
    return flagged_logs
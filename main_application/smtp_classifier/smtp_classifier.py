import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
loaded_model = joblib.load('smtp_classifier/smtp_classifier_model.pkl')
scaler = joblib.load('smtp_classifier/smtp_scaler.pkl')

def preprocess_smtp_logs(df):
    """
    Preprocess the input SMTP logs DataFrame to match the training format.
    """
    # Select only the required features (without labels since it's for classification)
    feature_list = [
        'domain_val_message-id', 'domain_match_message-id_from', 
        'domain_match_from_return-path', 'domain_match_message-id_return-path',
        'domain_match_message-id_sender', 'domain_match_message-id_reply-to', 
        'domain_match_return-path_reply-to', 'domain_match_reply-to_to', 
        'domain_match_to_in-reply-to', 'domain_match_errors-to_message-id', 
        'domain_match_errors-to_from', 'domain_match_errors-to_sender', 
        'domain_match_errors-to_reply-to', 'domain_match_sender_from', 
        'domain_match_references_reply-to', 'domain_match_references_in-reply-to', 
        'domain_match_references_to', 'domain_match_from_reply-to', 
        'domain_match_to_from', 'domain_match_to_message-id', 
        'domain_match_to_received'
    ]
    
    # Ensure the DataFrame contains only the features used in training
    df = df[feature_list]
    
    # Apply the scaler to standardize the features
    df_scaled = scaler.transform(df)
    
    return pd.DataFrame(df_scaled, columns=feature_list)

def smtp_classifier(smtp_logs):
    """
    Classifies SMTP logs, storing and returning phishing logs.
    
    Parameters:
    smtp_logs (pd.DataFrame): DataFrame containing SMTP logs in the same feature format.
    
    Returns:
    phishing_logs (list): List of logs classified as phishing.
    """
    # Preprocess the smtp logs
    X_new = preprocess_smtp_logs(smtp_logs)
    
    # Classify the logs
    predictions = loaded_model.predict(X_new)
    
    # Collect logs classified as phishing (label = 1)
    phishing_logs = smtp_logs[predictions == 1].to_dict(orient='records')
    
    return phishing_logs
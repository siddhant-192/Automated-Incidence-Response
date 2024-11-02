import json

def surikata_classifier(surikata_logs):
    flagged_logs = []
    for log in surikata_logs:
        if "ssh" in log:
            flagged_logs.append(log)
    
    return flagged_logs
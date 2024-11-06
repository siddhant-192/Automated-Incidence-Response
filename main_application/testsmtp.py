from smtp_classifier.smtp_classifier import smtp_classifier
import pandas as pd

df = pd.read_csv('smtp_test1.csv')

logs = smtp_classifier(df)

print(len(logs))

for log in logs:
    print(log)
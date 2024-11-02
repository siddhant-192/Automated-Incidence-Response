# This is the main orchestrating file for the project which handels all the processes and tasks.

# Importing the required libraries
from wahzu_classifier.wahzu_classifier import wahzu_classifier
from surikata_classifier.surikata_classifier import surikata_classifier
from smtp_classifier.smtp_classifier import smtp_classifier
from system_classifier.system_classifier import system_classifier
from llm.llm_client import generate_text

# Trigger function
def trigger():
    # All the logs will be retrieved from the multiple sources (Wahzu, Surikata, SMTP, System)
    wahzu_logs = [] # logs from Wahzu
    surikata_logs = [] # logs from Surikata
    smtp_logs = [] # logs from SMTP server
    system_logs = [] # logs from the system

    flagged_logs = classification(wahzu_logs=wahzu_logs, surikata_logs=surikata_logs, smtp_logs=smtp_logs, system_logs=system_logs)

    report = analyst_llm(flags=flagged_logs)

    markup_report = markup_report_llm(report=report)

    report_generator(markup_report=markup_report)

    print("Trigger function called")

def classification(wahzu_logs, surikata_logs, smtp_logs, system_logs):
    print("Classification function called")
    # Classification of logs from Wahzu
    wahzu_flagged = wahzu_classifier(wahzu_logs=wahzu_logs)

    # Classification of logs from Surikata
    # JSON data from Surikata will be passed
    surikata_flagged = surikata_classifier(surikata_logs=surikata_logs)

    # Classification of logs from SMTP server
    smtp_flagged = smtp_classifier(smtp_logs=smtp_logs)

    # Classification of logs from the system
    system_flagged = system_classifier(system_logs=system_logs)

    flagged_logs = {'WaZhu': wahzu_flagged, 'Surikata': surikata_flagged, 'SMTP': smtp_flagged, 'System': system_flagged}

    return flagged_logs

def analyst_llm(flags):
    print("LLM function called")
    # Make call to a LLM server setup to analyse and then generate report on the flagged logs
    # Prompt
    prompt = ""

    input_text = prompt + " ".join(flags.values())

    # Output report
    report = generate_text(input_text=input_text)
    return report

def markup_report_llm(report):
    print("Report Generation function called")
    # Generate report based on the analysis using another llm call in ASCII syntax
    # Prompt
    prompt = ""

    input_text = prompt + report
    # Output syntax doc
    markup_report = generate_text(input_text=input_text)
    return markup_report

def report_generator(markup_report):
    print("Report Generation function called")
    # Compile the syntax to make a report and downloadable in pdf format
    # Auto downloads the report to the user's system

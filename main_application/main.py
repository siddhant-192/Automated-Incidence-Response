# This is the main orchestrating file for the project which handels all the processes and tasks.

# Importing the required libraries
from wahzu_classifier.wahzu_classifier import wahzu_classifier
from surikata_classifier.surikata_classifier import surikata_classifier
from smtp_classifier.smtp_classifier import smtp_classifier
#from system_classifier.system_classifier import system_classifier
from pdf_maker import syntax_to_pdf
from llm.llm_client import generate_text
import pandas as pd
from rich.console import Console
from rich.text import Text
import time
import random
from PIL import Image
import numpy as np
import warnings
import sys
import os
from datetime import datetime

# Redirect warnings to null
warnings.simplefilter("ignore")
sys.stderr = open(os.devnull, "w")

# Load or run code that generates warnings here
# e.g., model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Reset stderr to default after loading
sys.stderr = sys.__stderr__

def initialisation_hackore():
    console = Console()

    # ASCII art for "ARTEMISCAN" main title
    ascii_art_title = """
     █████╗ ██████╗ ████████╗███████╗███╗   ███╗██╗███████╗ ██████╗ █████╗ ███╗   ██╗
    ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝████╗ ████║██║██╔════╝██╔════╝██╔══██╗████╗  ██║
    ███████║██████╔╝   ██║   █████╗  ██╔████╔██║██║███████╗██║     ███████║██╔██╗ ██║
    ██╔══██║██╔══██╗   ██║   ██╔══╝  ██║╚██╔╝██║██║╚════██║██║     ██╔══██║██║╚██╗██║
    ██║  ██║██║  ██║   ██║   ███████╗██║ ╚═╝ ██║██║███████║╚██████╗██║  ██║██║ ╚████║
    ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝                                                                                                                                                   
    """

    # Primary and secondary colors for alternating effect
    primary_color = "bold green"
    secondary_color = "green"

    # Display ASCII art with a typing effect using alternating colors
    def display_ascii_with_typing_effect(ascii_art, delay):
        toggle = True  # To alternate colors
        for char in ascii_art:
            color = primary_color if toggle else secondary_color
            console.print(char, style=color, end="")
            time.sleep(delay)
            toggle = not toggle if char != "\n" else toggle
        console.print()

    # Generate ASCII art from an image with specified width to match title
    ASCII_CHARS = "@%#*+=-:. "

    def generate_ascii_art(image_path, new_width=100):
        # Load and resize the image
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        # Adjust aspect ratio to make it slightly wider
        aspect_ratio = image.height / image.width
        new_height = int(new_width * aspect_ratio * 0.5)
        image = image.resize((new_width, new_height))
        
        # Convert to numpy array
        pixels = np.array(image)
        
        # Generate ASCII art with color
        ascii_art = Text()
        for row in pixels:
            for pixel in row:
                r, g, b = pixel  # RGB values
                brightness = int(0.3 * r + 0.59 * g + 0.11 * b)
                char = ASCII_CHARS[brightness // 32]
                ascii_art.append(char, style=f"rgb({r},{g},{b})")
            ascii_art.append("\n")
        
        return ascii_art

    # Fun hacker-style messages (memes) for a light-hearted vibe
    hacker_messages = [
        "[+] Initiating All systems and collectors....",
        "[+] Setting up the LLM mainframe... almost there!",
        "[+] Artemis is ready to be deployed!"
    ]

    # Color themes for the last three messages
    hacker_colors = ["bold cyan", "bold magenta", "bold yellow"]

    # Function to display hacker messages with typing effect and colored text for last three messages
    def display_hacker_messages(messages, delay=0.05):
        for i, message in enumerate(messages):
            # Choose a color for the last three messages
            if i >= len(messages) - 3:
                color = hacker_colors[i - (len(messages) - 3)]
            else:
                color = "bold green"  # Default color for initial messages
            
            # Display the message with typing effect
            hacker_text = Text(message, style=color)
            for char in hacker_text:
                console.print(char, end="")
                time.sleep(delay)
            console.print()  # Newline after message

    # Path to your image file
    image_path = "image.png"

    # Display the main title with typing effect
    display_ascii_with_typing_effect(ascii_art_title,0.006)

    # Generate and display ASCII art from the logo image
    ascii_art_image = generate_ascii_art(image_path, new_width=100)
    display_ascii_with_typing_effect(ascii_art_image,0.0005)

    # Display hacker-style messages with special colors for the last three messages
    display_hacker_messages(hacker_messages)

    # Prompt the user to press Enter to continue
    # console.print("\nPress [bold green]Enter[/bold green] to continue...", style="bold")
    # input()



    
# Trigger function
def trigger():
    # All the logs will be retrieved from the multiple sources (Wahzu, Surikata, SMTP, System)
    # WAHZU LOGS
    wahzu_logs = pd.read_csv('wahzu_test.csv') # logs from Wahzu
    print("Waazu logs loaded")

    # SURIKATA LOGS
    surikata_logs = pd.read_csv('surikata_test.csv') # logs from Surikata
    print("Surikata logs loaded")

    # SMTP LOGS
    smtp_logs = pd.read_csv('smtp_test.csv') # logs from SMTP server
    print("SMTP logs loaded")

    # SYSTEM LOGS
    #system_logs = [] # logs from the system

    flagged_logs = classification(wahzu_logs=wahzu_logs, surikata_logs=surikata_logs, smtp_logs=smtp_logs) #system_logs=system_logs
    print("Classification completed")

    print("Analysing...")
    report = analyst_llm(flags=flagged_logs)
    print("Analysis completed")

    print("Generating report...")
    markup_report_timestamp = markup_report_llm(report=report)
    print("Report generated")

    report_generator(timestamp=markup_report_timestamp)

def classification(wahzu_logs, surikata_logs, smtp_logs):
    # Classification of logs from Wahzu
    wahzu_flagged = wahzu_classifier(wahzu_logs=wahzu_logs)

    # Classification of logs from Surikata
    # JSON data from Surikata will be passed
    surikata_flagged = surikata_classifier(surikata_logs=surikata_logs)

    # Classification of logs from SMTP server
    smtp_flagged = smtp_classifier(smtp_logs=smtp_logs)

    # Classification of logs from the system
    #system_flagged = system_classifier(system_logs=system_logs)

    flagged_logs = {'Wazhu': wahzu_flagged, 'Surikata': surikata_flagged, 'SMTP': smtp_flagged} #, 'System': system_flagged

    return flagged_logs

def analyst_llm(flags):
    # Make call to a LLM server setup to analyse and then generate report on the flagged logs
    # Prompt
    prompt = """
    You are an expert cybersecurity analyst specializing in incident response. Please analyze the following flagged logs collected from multiple sources. Your tasks are:

    Analyze the Logs:
        Examine each log individually to identify any signs of security incidents.
        Correlate events across different logs to uncover patterns or connections that indicate a broader incident.

    Generate an Incident Response Report:
        Based on your analysis, create a comprehensive incident response report.
        Use the provided report structure as a guideline, but feel free to adjust it as necessary to suit the specifics of the incident(s).
        Ensure the report is clear, concise, and appropriate for both technical and non-technical stakeholders.

    Report Structure:

        Executive Summary
            Overview of the Incident
            Impact Assessment
            Actions Taken
            Current Status

        Introduction
            Purpose of the Report
            Scope
            Audience

        Incident Description
            Timeline of Events
            Detection Method
            Affected Systems and Data
            Type of Incident

        Detection and Analysis
            Logs Collected
            Analysis Procedures
            Findings
            Correlation of Events

        Response Actions
            Containment Measures
            Eradication Steps
            Recovery Process
            Communication

        Root Cause Analysis
            Underlying Cause
            Contributing Factors

        Impact Assessment
            Business Impact
            Data Loss or Exposure
            Regulatory Compliance Implications

        Lessons Learned
            What Worked Well
            Areas for Improvement
            Response Effectiveness

        Recommendations
            Preventive Measures
            Security Enhancements
            Training Needs

        Conclusion
            Summary of Incident and Response
            Next Steps

        Appendices (as needed)
            Supporting Evidence
            Technical Details
            Contact Information

    Notes:

        Flexibility: Adjust the report structure to best fit the incident(s) you're reporting on. If certain sections are not applicable, you may omit them or combine sections for clarity.
        Clarity and Detail: Provide sufficient technical details in the main body or appendices to support your findings and conclusions.
        Confidentiality: Do not include any sensitive information that is not relevant to the incident analysis.

        SUPER IMPORTANT: Only output the report content and nothing else apart form it.

    Flagged Logs:
    """

    # Convert each log list to a string
    logs_text = "\n".join([" ".join(map(str, flag)) if isinstance(flag, list) else str(flag) for flag in flags.values()])
    input_text = prompt + logs_text

    # Output report
    report = generate_text(input_text=input_text)
    return report

def markup_report_llm(report):
    # Generate report based on the analysis using another llm call in ASCII syntax
    # Prompt
    prompt = """
        You are an expert in document formatting with proficiency in AsciiDoc syntax. Your task is to convert the following incident response report into AsciiDoc format. Please adhere strictly to the following guidelines:

        Conversion Requirements:

        - Headings:
        - Use a single equals sign (`=`) for the main title.
        - Use double equals signs (`==`) for section titles.
        - Use triple equals signs (`===`) for subsection titles, and so on.

        - Paragraphs:
        - Separate paragraphs with a blank line.

        - Lists:
        - Unordered Lists: Start items with an asterisk followed by a space (` `).
        - Ordered Lists: Start items with a period followed by a space (`. `).

        - Text Formatting:
        - Bold Text: Enclose text with double asterisks (`bold text`).
        - Italic Text: Enclose text with underscores (`_italic text_`).

        - Code Blocks:
        - Use four consecutive hyphens before and after the code block:
            ```Your code here```

        - Tables:
        - Use the AsciiDoc table syntax for any tabular data.

        - Images:
        - If there are images, use the syntax:
            ```image::filename.ext[Alt Text]```

        Output Instructions:

        - Exclusive Content: Only output the AsciiDoc code for the converted report.
        - No Additional Text: Do not include any explanations, comments, or text other than the AsciiDoc code.
        - Compilation Ready: Ensure the AsciiDoc code is properly formatted and ready for compilation without any modifications.

        Incident Response Report:

        """

    input_text = prompt + report
    # Output syntax doc
    markup_report = generate_text(input_text=input_text)

    # Get the current time and format it as a string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save markup report to 'test.adoc'
    with open(f'report/report_{timestamp}.adoc', 'w') as file:
        file.write(markup_report)

    return timestamp

def report_generator(timestamp):
    # Compile the syntax to make a report and downloadable in pdf format

    syntax_to_pdf(timestamp=timestamp)

    print("Report generated successfully")


if __name__ == "__main__":
    initialisation_hackore()
    trigger()
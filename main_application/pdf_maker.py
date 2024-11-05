import subprocess
import pdfkit
from PyPDF2 import PdfMerger


def syntax_to_pdf(timestamp):

    # Run the asciidoc command
    result = subprocess.run(['asciidoc', f'report/report_{timestamp}.adoc'], capture_output=True, text=True)

    # Path to the HTML and output PDF files
    input_html = f'report/report_{timestamp}.html'
    output_pdf = f'report/report_{timestamp}.pdf'

    options = {
        'quiet': '',
        'no-outline': None,  # Remove outlines that may cause the issue
    }
    pdfkit.from_file(input_html, output_pdf, options=options)


    # Create a PdfMerger object
    merger = PdfMerger()

    # Append the PDFs to the merger in the desired order
    with open('report_cover_page.pdf', 'rb') as pdf1, open(f'report/report_{timestamp}.pdf', 'rb') as pdf2:
        merger.append(pdf1)
        merger.append(pdf2)

    # Write the merged PDF to a new file
    with open(f'report/incident_report_{timestamp}.pdf', 'wb') as merged_pdf:
        merger.write(merged_pdf)
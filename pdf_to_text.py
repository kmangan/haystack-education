import os
from PyPDF2 import PdfReader

# Simple utility to convert the python PDFs downloaded from https://docs.python.org/3/ to text files.
# Note that the Haystack pipeline in haystack_setup.py can work with PDFs or Text files though.
def convert_pdf_to_text(pdf_path, output_dir):

    # Extract the PDF file name (without extension)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Output text file path
    output_file = os.path.join(output_dir, f"{base_name}.txt")
    
    try:
        # Read the PDF
        reader = PdfReader(pdf_path)
        with open(output_file, "w", encoding="utf-8") as text_file:
            for page in reader.pages:
                text_file.write(page.extract_text() + "\n")
        print(f"Converted '{pdf_path}' to '{output_file}'.")
    except Exception as e:
        print(f"Error processing '{pdf_path}': {e}")

def batch_convert_pdfs(pdf_dir, output_dir):
    """
    Converts all PDFs in a directory to text files.
    """
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file_name)
            convert_pdf_to_text(pdf_path, output_dir)

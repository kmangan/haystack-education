import os
from pdf_to_text import batch_convert_pdfs

def main():
    """
    Main function to convert PDFs to text files.
    """
    pdf_dir = "data/raw"          # Directory containing input PDFs
    output_dir = "data/processed" # Directory to save the text files

    # Ensure the directories exist
    if not os.path.exists(pdf_dir):
        print(f"Error: Input directory '{pdf_dir}' does not exist. Please add your PDFs there.")
        return

    # Start PDF-to-text conversion
    print(f"\nConverting PDFs from '{pdf_dir}' to text files in '{output_dir}'...\n")
    try:
        batch_convert_pdfs(pdf_dir, output_dir)
        print("\nPDF conversion completed")
    except Exception as e:
        print(f"\nAn error occurred during the conversion process: {e}")

if __name__ == "__main__":
    main()

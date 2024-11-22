from haystack.document_stores.in_memory import InMemoryDocumentStore
import os

def load_text_files_to_documents(data_dir):
    """
    Load text files from a directory and convert them to Haystack document format.
    
    :param data_dir: Path to the directory containing text files.
    :return: List of documents in Haystack format.
    """
    documents = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            documents.append({
                "content": content,
                "meta": {"file_name": file_name}  # Add metadata (e.g., file name)
            })
    return documents

def initialize_document_store(data_dir):
    """
    Initialize the document store and load documents from text files.
    
    :param data_dir: Path to the directory containing text files.
    :return: InMemoryDocumentStore instance.
    """
    # Create the document store
    document_store = InMemoryDocumentStore()

    # Load text files and write documents to the store
    documents = load_text_files_to_documents(data_dir)
    document_store.write_documents(documents)
    print(f"Loaded {len(documents)} documents into the document store.")
    return document_store

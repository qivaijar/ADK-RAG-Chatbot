from dotenv import load_dotenv
from google.cloud import storage
import os

load_dotenv()
storage_client = storage.Client()


def upload_doc(file_path: str) -> str:
    """
    Upload new documents.

    Args:
        file_paths (str): The paths of the file that will be added into bucket (for RAG knowledge base).

    Returns:
        status (str): the status of the document upload process.
    """
    try:
        doc_bucket = storage_client.bucket(os.getenv("DOC_BUCKET"))
        file_name = os.path.basename(file_path)
        blob = doc_bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        return f"All files have been uploaded to {doc_bucket}"

    except Exception as e:
        return f"An error occured during upload: {e}"


def list_docs() -> list[str]:
    """
    Upload new documents.

    Returns:
        status (list[str]): the list of document files inside the bucket (for RAG knowledge base).
    """
    try:
        doc_bucket = storage_client.get_bucket(os.getenv("DOC_BUCKET"))
        doc_list = [x.name[:-4] for x in doc_bucket.list_blobs()]
        return doc_list

    except Exception as e:
        return f"Cannot list documents: {e}"

import pandas as pd
import requests
import yaml
import os
import time
import json
import pdfplumber

from google.genai.errors import ClientError
from google.genai.types import EmbedContentConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import storage, aiplatform
from google import genai


# Define global variables
load_dotenv()
storage_client = storage.Client()
genai_client = genai.Client(
    vertexai=True,
    project=os.getenv('GOOGLE_CLOUD_PROJECT'),
    location=os.getenv('GOOGLE_SERVICE_LOCATION')
)

chunk_size = 600
chunk_overlap = 80
langchain_splitter = splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_LOCATION'))

index = aiplatform.MatchingEngineIndex(index_name=os.getenv('INDEX_URL'))


# Define tools and supporting functions
def upload_doc(file_path: str) -> str:
    """
    Upload new document to the bucket.

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
    List the documents inside the bucket.

    Returns:
        doc_list (list[str]): the list of document files inside the bucket (for RAG knowledge base).
    """
    try:
        doc_bucket = storage_client.get_bucket(os.getenv("DOC_BUCKET"))
        doc_list = [x.name for x in doc_bucket.list_blobs()]
        return doc_list

    except Exception as e:
        return f"Cannot list documents: {e}"


def delete_doc(doc_name: str) -> str:
    """
    Delete a document from the bucket (for RAG knowledge base).

    Args:
        doc_name: name of the document to be deleted.

    Returns:
        status (str): the status of the document deletion process.
    """
    try:
        doc_bucket = storage_client.bucket(os.getenv("DOC_BUCKET"))
        doc = doc_bucket.blob(doc_name)
        doc.delete()
        return f"Document {doc_name} has been deleted from the bucket."

    except Exception as e:
        return f"Cannot delete document: {e}"


def _extract_batch_embeddings(text_chunks: list[str], batch_size=8, retry_limit=5):
    embeddings = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i: i + batch_size]

        for attempt in range(retry_limit):
            try:
                response = genai_client.models.embed_content(
                    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
                    contents=batch,
                    config=EmbedContentConfig(
                        output_dimensionality=3072
                    )
                )
                batch_embeddings = [e.values for e in response.embeddings]
                embeddings.extend(batch_embeddings)
                time.sleep(2)  # try to avoid quota limit
                break

            except ClientError as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    wait = 2 ** attempt
                    print(f"Quota hit. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise e
    return embeddings


def _write_jsonl(input_df, emb_filename):
    target_df = input_df[['id', 'embedding']]
    target_df['embedding_metadata'] = input_df.apply(
        lambda row: {
            "text_chunk": row['text_chunk'],
            "source": row['source']
        },
        axis=1
    )

    with open(emb_filename, 'a') as f:
        for index, row in target_df.iterrows():
            json_line = json.dumps(row.to_dict())
            f.write(json_line + '\n')


def _get_web_embeddings(emb_filename: str):
    with open("rag_chatbot/web_scrap_list.yaml", "r") as f:
        faq_links = yaml.safe_load(f)

    id_count = 1
    data_list = []

    for link in faq_links["links"]:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        questions = soup.find_all("div", class_="ewd-ufaq-faq-title-text")
        answers = soup.find_all(
            "div", class_="ewd-ufaq-post-margin ewd-ufaq-faq-post")

        for q, a in zip(questions, answers):
            q, a = q.get_text(strip=True), a.get_text(strip=True)
            id = f"web-{id_count}"
            text_chunk = f"Q: {q}, A: {a}"
            source = link

            data_list.append(
                {"id": id, "text_chunk": text_chunk, "source": source})

            id_count += 1

    web_dataframe = pd.DataFrame(data_list)
    text_embeddings = _extract_batch_embeddings(
        text_chunks=web_dataframe["text_chunk"].to_list()
    )
    web_dataframe["embedding"] = text_embeddings

    # Write to json File
    _write_jsonl(web_dataframe, emb_filename)


def _process_csv(csv_file: str, id_count: int):
    pd_csv = pd.read_csv(csv_file)
    data_list = []
    for index, row in pd_csv.iterrows():
        id = f"doc-{id_count}"
        id_count += 1
        text_chunk = f"Q: {str(row['Question']).strip()}, A: {
            str(row['Response']).strip()}"
        source = csv_file

        # Append to data list
        data_list.append(
            {
                'id': id,
                'text_chunk': text_chunk,
                'source': source
            }
        )

    return pd.DataFrame(data_list), id_count


def _process_txt(txt_file: str, id_count: int):
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()
    data_list = []
    chunks = langchain_splitter.split_text(text)
    for chunk in chunks:
        id = f"doc-{id_count}"
        id_count += 1
        text_chunk = chunk.strip()
        source = txt_file

        # Append to data list
        data_list.append(
            {
                'id': id,
                'text_chunk': text_chunk,
                'source': source
            }
        )
    return pd.DataFrame(data_list), id_count


def _process_pdf(pdf_file: str, id_count: int):

    data_list = []

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):

            texts = page.extract_text()
            tables = page.extract_table()

            # Process texts
            if texts:
                text_chunks = langchain_splitter.split_text(texts)
                for chunk in text_chunks:
                    id = f"doc-{id_count}"
                    id_count += 1
                    text_chunk = chunk.strip()
                    source = pdf_file

                    # Append to data list
                    data_list.append(
                        {
                            'id': id,
                            'text_chunk': text_chunk,
                            'source': source
                        }
                    )
            # Process tables
            if tables:
                table_df = pd.DataFrame(
                    tables[1:], columns=tables[0])
                table_markdown = table_df.to_markdown(index=False)
                id = f"doc-{id_count}"
                id_count += 1
                text_chunk = table_markdown
                source = pdf_file

                # Append to data list
                data_list.append(
                    {
                        'id': id,
                        'text_chunk': text_chunk,
                        'source': source
                    }
                )

    return pd.DataFrame(data_list), id_count


def _get_doc_embeddings(emb_filename: str):
    # Create an empty document dataframe
    doc_dataframe = pd.DataFrame(columns=['id', 'text_chunk', 'source'])
    id_count = 0
    bucket = storage_client.bucket(os.getenv('DOC_BUCKET'))
    bucket_docs = list_docs()

    for file in bucket_docs:
        # Download file to local machine
        doc = bucket.blob(file)
        doc.download_to_filename(file)

        # Start processing
        file_extension = file[-4:]
        if file_extension == '.csv':
            result_dataframe, id_count = _process_csv(file, id_count)
        elif file_extension == '.txt':
            result_dataframe, id_count = _process_txt(file, id_count)
        elif file_extension == '.pdf':
            result_dataframe, id_count = _process_pdf(file, id_count)

        # Delete the file after processing
        os.remove(file)

        # Append dataframe
        doc_dataframe = pd.concat(
            [doc_dataframe, result_dataframe], ignore_index=True)

    text_embeddings = _extract_batch_embeddings(
        text_chunks=doc_dataframe["text_chunk"].to_list()
    )
    doc_dataframe["embedding"] = text_embeddings

    # Write to json File
    _write_jsonl(doc_dataframe, emb_filename)


def update_rag_knowledge() -> str:
    """
    Update the RAG knowledge base

    Returns:
        status (str): the status of the RAG knowledge base update process.
    """
    # Delete the currently existing embedding file
    emb_bucket = storage_client.bucket(os.getenv("EMB_BUCKET"))
    emb_file = emb_bucket.blob(os.getenv("EMB_FILENAME"))
    try:
        emb_file.delete()
    except Exception:
        pass

    # Create a new embedding json file
    emb_filename = os.getenv("EMB_FILENAME")
    if os.path.exists(emb_filename):
        os.remove(emb_filename)

    # Create embeddings from web knowledge
    _get_web_embeddings(emb_filename)
    # Create embeddings from document knowledge (bucket)
    _get_doc_embeddings(emb_filename)

    # Upload to embedding bucke
    emb_file.upload_from_filename(emb_filename)

    # Update index
    gcs_uri = f"gs://{os.getenv('EMB_BUCKET')}/{os.getenv('EMB_FILENAME')}"
    index.update_embeddings(
        contents_delta_uri=gcs_uri, is_complete_overwrite=True
    )

    return "RAG Knowledge base has been updated."


def generate_rag_answer(user_query: str) -> str:
    """
    Answer user's query based on the RAG knowledge base

    Args:
        user_query (str): user's question related to the knowledge base.

    Returns:
        answer (str): generated answer based on the retrived contexts.
    """
    # Define system prompt
    # Get query embeddings
    # Fetch nearest neighbors
    # Rerank results
    # Generate answers using vllm model
    pass

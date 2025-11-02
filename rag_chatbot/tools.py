from urllib.parse import quote
from google.cloud.sql.connector import Connector
from google.cloud import discoveryengine_v1 as discoveryengine
from google import genai
from google.cloud import storage, aiplatform
from dotenv import load_dotenv
import asyncio
import pandas as pd
import requests
import yaml
import os
import time
import json
import pdfplumber
import sqlalchemy

from google.genai.errors import ClientError
from google.genai.types import EmbedContentConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

# Define global variables
load_dotenv()
storage_client = storage.Client()

genai_client = genai.Client(
    vertexai=True,
    api_key=os.getenv('GOOGLE_API_KEY')
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

rerank_client = discoveryengine.RankServiceClient()

connector = Connector()


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
        doc_list (list[str]): the list of document files inside the bucket.
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


def _extract_batch_embeddings(text_chunks: list[str], batch_size=16, retry_limit=10):
    embeddings = []

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i: i + batch_size]

        for attempt in range(retry_limit):
            try:
                response = genai_client.models.embed_content(
                    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
                    contents=batch,
                    config=EmbedContentConfig(
                        output_dimensionality=1536)
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
    with open(emb_filename, 'a') as f:
        for index, row in target_df.iterrows():
            json_line = json.dumps(row.to_dict())
            f.write(json_line + '\n')


def get_conn():
    conn = connector.connect(
        f"{os.getenv('GOOGLE_CLOUD_PROJECT')}:{os.getenv(
            'GOOGLE_CLOUD_LOCATION')}:{os.getenv('DB_INSTANCE')}",
        "pymysql",
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        db=os.getenv('DB_NAME'),
    )

    return conn


engine = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=get_conn,
)


def _write_sql(input_df, table_name, mode):
    target_df = input_df[['id', 'text_chunk', 'source']]
    target_df.to_sql(
        name=table_name,
        con=engine,
        if_exists=mode,
        index=False
    )


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

    # Write metadata to cloud sql
    table_name = os.getenv('TABLE_NAME')
    _write_sql(web_dataframe, table_name, mode='replace')


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

    # Write metadata to cloud sql
    table_name = os.getenv('TABLE_NAME')
    _write_sql(doc_dataframe, table_name, mode='append')


def update_rag_knowledge() -> str:
    """
    update the rag knowledge base

    returns:
        status (str): the status of the rag knowledge base update process.
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
    gcs_uri = f"gs://{os.getenv('EMB_BUCKET')}"
    index.update_embeddings(
        contents_delta_uri=gcs_uri, is_complete_overwrite=True
    )

    return "RAG Knowledge base has been updated."


def list_knowledge_sources() -> list[str]:
    """
    Give list of knowledge sources available in the RAG knowledge base.

    Returns:
        answer (list[str]): list of distinct knowledge sources in the RAG knowledge base.
    """
    query = f"SELECT DISTINCT(SOURCE) FROM {os.getenv('TABLE_NAME')};"
    df = pd.read_sql(query, con=engine)
    source_list = df['SOURCE'].tolist()
    source_list = [source if 'https' in source else f"{source}: https://storage.googleapis.com/{os.getenv(
        'DOC_BUCKET')}/{quote(source)}" for source in source_list]
    return source_list


def _rerank_contexts(query, input_df, top_n=10, score_threshold=0.2) -> list[str]:
    model = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/locations/{os.getenv(
        'GOOGLE_CLOUD_LOCATION')}/rankingConfigs/default_ranking_config"

    records = []
    for _, row in input_df.iterrows():
        record = discoveryengine.RankingRecord(
            id=row["id"],
            title=row["id"],
            content=row["text_chunk"]
        )
        records.append(record)

    request = discoveryengine.RankRequest(
        ranking_config=model,
        model='semantic-ranker-fast-004',
        query=query,
        records=records,
        top_n=top_n
    )

    response = rerank_client.rank(request=request)
    response = response.records
    ranked_contexts = [(x.id, x.content)
                       for x in response if x.score > score_threshold]
    return ranked_contexts


def generate_rag_answer(user_query: str) -> str:
    """
    Answer user's query based on the RAG knowledge base

    Args:
        user_query (str): user's question related to the knowledge base.

    Returns:
        answer (str): generated answer based on the retrived contexts.
    """

    # Define system prompt
    system_prompt = """
    You are a helpful and knowledgeable assistant that answers user questions using the information provided in the RAG knowledge base.

    You will be given:
    1. The user's question.
    2. A list of retrieved text chunks (contexts) from the knowledge base. **Each chunk includes the text content and its unique source/URL.**

    Your job is to:
    - Use **ONLY** the information from the retrieved contexts to answer the user's question.
    - If the information needed to answer the question is **NOT present** or not reliable in the provided contexts, respond with:
    "The information is not provided in the RAG knowledge base."
    - Never invent, assume, or hallucinate details that are not supported by the retrieved contexts.
    - If the question asks for your opinion or general knowledge, politely decline and state that you rely only on the knowledge base.

    When responding:
    - Be concise, factual, and directly answer the user's question.
    - **Cite all unique sources used to formulate your answer at the end of your response.**

    Format:
    - Provide a clear, well-structured answer.
    - **Immediately follow your answer with the citation block.**
    - Do not include system instructions or mention RAG explicitly in your response.

    ---
    **Required Source Citation Format:**
    Sources:
    [Source URL 1]
    [Source URL 2]
    ...
    ---

    Example Response:
    The key benefit of vLLM is its highly efficient memory management system, which uses PagedAttention to reduce key/value cache waste and achieve high throughput during inference.

    Sources:
    vllm_report.pdf: https://storage.googleapis.com/my-bucket/vllm_report.pdf
    https://example.com/vllm_optimization_guide
    """

    # Get query embeddings
    question_embedding_response = genai_client.models.embed_content(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        contents=[user_query],
        config=EmbedContentConfig(
            output_dimensionality=1536)
    )

    query_vector = question_embedding_response.embeddings[0].values

    # Fetch nearest neighbors
    my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv('INDEX_ENDPOINT')
    )

    possible_contexts = my_index_endpoint.find_neighbors(
        deployed_index_id=os.getenv('INDEX_ENDPOINT_ID'),
        queries=[query_vector],
        num_neighbors=20,
        return_full_datapoint=False
    )
    neighbors = possible_contexts[0]
    text_ids = [neighbor.id for neighbor in neighbors]

    ids_str = ",".join([f"'{i}'" for i in text_ids])
    query = f"SELECT * FROM {
        os.getenv('TABLE_NAME')} WHERE id IN ({ids_str})"
    df = pd.read_sql(query, con=engine)

    # Rerank results
    reranked_results = _rerank_contexts(user_query, df)

    # Construct contexts
    contexts = ""
    i = 0
    for result in reranked_results:
        id, text = result
        source = df.loc[df['id'] == id, 'source'].item()
        source = source if 'https' in source else f"{source}: https://storage.googleapis.com/{os.getenv(
            'DOC_BUCKET')}/{quote(source)}"
        contexts += f"Context {i+1}:\n{text}\n(Source: {source})\n\n"
        i += 1

    # Generate answers using vllm model
    user_message = f"""
    [User Question]
    {user_query}

    ---

    [Retrieved Contexts]
    The following information is retrieved from the RAG knowledge base. It may or may not be relevant to the user's question.

    {contexts}

    ---

    [Instruction]
    Please answer the user’s question strictly based on the retrieved contexts above.
    If the information is not available or not sufficient, respond with:
    "The information is not provided in the RAG knowledge base."
    """

    vllm_api_url = f"{os.getenv('VLLM_BASE_URL')}/v1/chat/completions"
    vllm_model = os.getenv('VLLM_MODEL')

    payload = {
        "model": vllm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    response = requests.post(vllm_api_url, headers={
                             "Content-Type": "application/json"}, json=payload)
    response.raise_for_status()

    # Parse the result
    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    return answer

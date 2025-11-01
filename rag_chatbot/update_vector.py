from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pandas as pd
import requests
import os
import yaml
import tempfile
from tqdm import tqdm
import time
from google import genai
from google.genai.errors import ClientError


load_dotenv()


def extract_batch_embedding(text_chunks: list[str], genai_client, batch_size=2, retry_limit=5):
    embeddings = []

    for i in tqdm(range(0, len(text_chunks), batch_size)):
        batch = text_chunks[i: i + batch_size]

        for attempt in range(retry_limit):
            try:
                response = genai_client.models.embed_content(
                    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
                    contents=batch
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


def update_rag_knowledge_web():
    client = genai.Client(
        vertexai=True,
        project=os.getenv('GOOGLE_CLOUD_PROJECT'),
        location=os.getenv('GOOGLE_SERVICE_LOCATION')
    )

    with open("web_scrap_list.yaml", "r") as f:
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

    web_dataframe = pd.DataFrame(data_list).set_index("id")
    text_embeddings = extract_batch_embedding(
        text_chunks=web_dataframe["text_chunk"].to_list(),
        genai_client=client
    )
    web_dataframe["text_embeddings"] = text_embeddings
    breakpoint()


if __name__ == "__main__":
    update_rag_knowledge_web()

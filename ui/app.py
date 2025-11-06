import gradio as gr
import uuid
import yaml
import requests
from dotenv import load_dotenv
from google.cloud import storage
import os

storage_client = storage.Client()


def upload_docs(file_paths: list[str]) -> str:
    try:
        doc_bucket = storage_client.bucket(os.getenv("DOC_BUCKET"))
        for file in file_paths:
            file_name = os.path.basename(file)
            blob = doc_bucket.blob(file_name)
            blob.upload_from_filename(file)
    except Exception as e:
        return f"An error occured during upload: {e}"


def send_query(new_message: str):
    url = config["send_query_url"].format(
        host=os.getenv("AGENT_HOST")
    )

    headers = {"Content-Type": "application/json"}

    body = {
        "app_name": config["app_name"],
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {"role": "user", "parts": [{"text": new_message}]},
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    response = response.json()
    try:
        answer = response[0]["content"]["parts"][0]["text"]
    except Exception:
        answer = response[-1]["content"]["parts"][0]["text"]
    return answer


def create_user_session_ids():
    user_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
    response = requests.post(
        config["user_session_url"].format(
            host=os.getenv("AGENT_HOST"),
            app_name=config["app_name"],
            user_id=user_id,
            session_id=session_id,
        )
    )

    response.raise_for_status()
    return user_id, session_id


def answer_question(message, history, files):
    if files is not None:
        upload_docs(files)
        return "File(s) has been uploaded into the document bucket!"
    result = send_query(message)
    return result


if __name__ == "__main__":
    # Initialize variables
    load_dotenv()
    global config
    with open("ui_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create user & session ids
    global user_id
    global session_id
    user_id, session_id = create_user_session_ids()

    # Define ui
    demo = gr.ChatInterface(
        fn=answer_question,
        textbox=gr.Textbox(
            placeholder="Type your query here...",
            label="Input query",
        ),
        additional_inputs=[
            gr.Files(label="Upload a file to add it to the RAG knowledge base.")
        ],
        title="Google ADK RAG Chatbot",
        description="An Chatbot implementation of RAG-based system using Google ADK, developed by Rizki Rivai Ginanjar.",
    )

    demo.launch(share=False, show_api=False)

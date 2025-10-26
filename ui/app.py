import gradio as gr
import uuid
import yaml
import requests
from dotenv import load_dotenv
import os


def send_query(new_message: str):
    url = config['send_query_url'].format(
        host=os.getenv('AGENT_HOST'),
        port=os.getenv('AGENT_PORT')
    )

    headers = {"Content-Type": "application/json"}

    body = {
        "app_name": config['app_name'],
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [
                {
                    "text": new_message
                }
            ]
        }
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    response = response.json()
    answer = response[0]["content"]["parts"][0]["text"]
    return answer


def create_user_session_ids():
    user_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
    response = requests.post(config['user_session_url'].format(
        host=os.getenv('AGENT_HOST'),
        port=os.getenv('AGENT_PORT'),
        app_name=config['app_name'],
        user_id=user_id,
        session_id=session_id
    ))

    response.raise_for_status()
    return user_id, session_id


def answer_question(message, history, files):
    if files is not None:
        message.append("file_paths: {files}")
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
        description="An Chatbot implementation of RAG-based system using Google ADK, developed by Rizki Rivai Ginanjar."
    )

    demo.launch(share=False, show_api=False)

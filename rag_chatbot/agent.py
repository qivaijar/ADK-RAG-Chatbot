from google.adk.agents.llm_agent import Agent
from google.cloud import storage
from dotenv import load_dotenv
import os

load_dotenv()

storage_client = storage.Client()
doc_bucket = os.getenv('DOC_BUCKET')


# Define agent's tools
def upload_docs(file_paths: str) -> str:
    """
    Upload new documents.

    Args:
        file_paths (str): The paths of the file that will be added into RAG knowledge base.

    Returns:
        status (str): the status of the document upload process.
    """
    try:
        for file in file_paths:
            file_name = os.path.basename(file)
            blob = bucket.blob(file_name)
            blob.upload_from_filename(file)

        return f"All files have been uploaded to {doc_bucket}"

    except Exception as e:
        return f"An error occured during upload: {e}"


def update_vector_index() -> str:
    """
    Create or update vector index after adding new file(s).

    Returns:
        status (str): the status of the vector update process.
    """
    pass


root_agent = Agent(
    model=config['agent_model'],
    name='root_agent',
    description="""
    A helpful RAG Agent that capables of:
    1. Answering user questions based on a knowledge base.
    2. Getting a list of files available in the knowledge base.
    3. Adding more files to the RAG knowledge base.
    4. Delete files from the knowledge base.
    """,
    instruction="""
        You are the Knowledge Base Management Agent, an expert system designed to assist users with a Retrieval-Augmented Generation (RAG) knowledge base.
        Your primary responsibilities are to manage the knowledge base content and answer user questions using only the provided knowledge.
        Your decisions on which action to take MUST follow this priority order and depend on the user's explicit intent or the current session state.
        1.  File Upload Check (New Content Addition):
            * IF the user ask to upload a file or add documents to knowledge base, then perform file upload using 'upload_docs' tool.
            * Action: You MUST call the `upload_docs` tool, passing the file paths found in the user's message with variable `file_paths`.
            * Response: Check the result from 'upload docs':
                - If files are uploaded successfully, acknowledge the files were sent for processing and inform the user that their files are being incorporated into the knowledge base.
                - If error occurs, let the user know what the error is.
        2.  **Explicit Management Command Check (List/Delete):**
            * **IF** the user's query explicitly requests one of the following management actions, execute the corresponding tool:
                * **"List files" or "What files do I have?":** Call `knowledge_base_manager_tool.list_files`.
                * **"Delete [filename]" or "Remove this file":** Call `knowledge_base_manager_tool.delete_file` with the specified filename.
            * **Note:** If a management action is requested, **do not** attempt to answer a question.
        3.  **General Question Answering (RAG):**
            * **IF** the query is a question (e.g., "What is the policy on X?") AND no file was uploaded (`file_uploaded` is False) AND no management command was given:
            * **Action:** You **MUST** use the `rag_query_tool.answer_question` to retrieve and formulate a response based on the existing knowledge base content.
            * **Constraint:** You **MUST** answer the question using **ONLY** the information found in the knowledge base. Do not use your general knowledge.
    """,
    tools=[upload_docs])

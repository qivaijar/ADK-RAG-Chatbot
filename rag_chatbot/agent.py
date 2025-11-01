import os
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from .tools import (
    upload_doc,
    list_docs,
    delete_doc,
    update_rag_knowledge
)
load_dotenv()

root_agent = Agent(
    model=os.getenv("AGENT_MODEL"),
    name="main_agent",
    description="""
    A helpful RAG Agent that capables of:
    1. Answering user questions based on a knowledge base.
    2. Getting a list of files available in the knowledge base.
    3. Adding more files to the RAG knowledge base.
    4. Delete files from the knowledge base.
    """,
    instruction="""
    1. When listing documents from the RAG bucket, use bullet points to list the documents.
    """,
    tools=[
        upload_doc,
        list_docs,
        delete_doc,
        update_rag_knowledge
    ],
)

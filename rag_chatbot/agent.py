import os
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from .tools import (
    upload_doc,
    list_docs,
    delete_doc,
    update_rag_knowledge,
    list_knowledge_sources,
    generate_rag_answer

)
load_dotenv()

root_agent = Agent(
    model=os.getenv("AGENT_MODEL"),
    name="root_agent",
    description="""
    A helpful RAG Agent that capables of:
    1. Answering user questions based on a knowledge base.
    2. Getting a list of sources available in the knowledge base.
    3. Adding more files to the RAG knowledge base (bucket only, not web source).
    4. Delete files from the knowledge base (bucket).
    """,
    instruction="""
    1. Use list_docs if the user ask what is inside the document bucket. use list_knowledge_sources if the user asks about the the list of sources inside the RAG knowledge base.
    2. When listing documents from the bucket or knowledge sources, use bullet points to list the documents.
    3. When the user ask to update the RAG knowledge base, just use the update_rag_knowledge tool, and wait until finished, then inform the user.
    4. Use generate_rag_answer to find the answer of the question from RAG knowledge base.
    5. After using generate_rag_answer, DO NOT modify the answer before sending it back to the user. Use the generated answer as it is and send the answer to the user.
    """,
    tools=[
        upload_doc,
        list_docs,
        delete_doc,
        update_rag_knowledge,
        list_knowledge_sources,
        generate_rag_answer
    ],
)

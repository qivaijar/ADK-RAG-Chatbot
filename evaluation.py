import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
import vertexai
from dotenv import load_dotenv
import os
from google.cloud import aiplatform
from rag_chatbot.tools import _rerank_contexts, generate_rag_answer
import pandas as pd
import sqlalchemy
from google import genai
from google.genai.types import EmbedContentConfig
from google.cloud.sql.connector import Connector
from urllib.parse import quote


load_dotenv('rag_chatbot/.env')

genai_client = genai.Client(
    vertexai=True,
    api_key=os.getenv('GOOGLE_API_KEY')
)
vertexai.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('GOOGLE_CLOUD_LOCATION'))
aiplatform.init(project=os.getenv('GOOGLE_CLOUD_PROJECT'),
                location=os.getenv('GOOGLE_CLOUD_LOCATION'))
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=os.getenv('INDEX_ENDPOINT')
)
connector = Connector()

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

def prepare_data():
    questions = [
        "How do I navigate the package summary",
        "What kind of case can vPost FFPs assist with?",
        "How do I navigate the address section on the portal",
        "Is bundling discount available?",
        "Which team can provide assistance for unclaimed vPost packages",
        "What do I do if my shipment goes to my old address?",
        "What is actual weight?",
        "How will I know that my package has arrived at our overseas warehouse?",
        "Why are personal effects and used household items not approved for shipping to Singapore?",
        "Does vPost provide guarantee on product authenticity?"
    ]

    data = []
    for question in questions:
        question_embedding_response = genai_client.models.embed_content(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
        contents=[question],
        config=EmbedContentConfig(
            output_dimensionality=1536)
        )

        query_vector = question_embedding_response.embeddings[0].values
        
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
        reranked_results = _rerank_contexts(question, df)
    
        # Construct contexts
        contexts = []
        i = 0
        for result in reranked_results:
            id, text = result
            source = df.loc[df['id'] == id, 'source'].item()
            source = source if 'https' in source else f"{source}: https://storage.googleapis.com/{os.getenv(
                'DOC_BUCKET')}/{quote(source)}"
            contexts.append(f"Context {i+1}:\n{text}\n(Source: {source})\n\n")
            i += 1
        
        answer = generate_rag_answer(question)

        data.append(
            {
                "question": question,
                "contexts": contexts,
                "answer": answer
            }
        )

    return data



if __name__ == "__main__":
    # Instantiate the evaluator LLM + embeddings
    evaluator_llm = LangchainLLMWrapper(
        VertexAI(model_name=os.getenv('AGENT_MODEL'))  # or another available model
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(model_name=os.getenv('GOOGLE_EMBEDDING_MODEL'), project=os.getenv('GOOGLE_CLOUD_PROJECT'), location=os.getenv('GOOGLE_CLOUD_LOCATION'))
    )

    data = prepare_data()

    # Create a Dataset
    dataset = Dataset.from_list(data)

    # Choose your metrics
    metrics = [faithfulness, answer_relevancy]

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print(result)

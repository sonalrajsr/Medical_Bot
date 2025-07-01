from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv()

os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

def store_vector_db(text_chunks):
    """Store text chunks in Pinecone vector database."""
    vector_store = PineconeVectorStore.from_documents(
        index_name = 'medical-chat-bot',
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        documents = text_chunks
    )
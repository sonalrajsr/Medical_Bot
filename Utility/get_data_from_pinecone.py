from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def get_result_pinecone(query):
    """Retrieve documents from Pinecone based on the query."""
    doc_search = PineconeVectorStore.from_existing_index(
        index_name='medical-chat-bot',
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    retriver_pinecone = doc_search.as_retriever(search_kwargs={"k": 5}, search_type="similarity")

    retrieved_docs = retriver_pinecone.invoke(query)
    top_matched_docs = ''
    for doc in retrieved_docs:
        top_matched_docs += doc.page_content + '\n\n'
    return top_matched_docs

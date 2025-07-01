from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(pages, chunk_size=500, chunk_overlap=50):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(pages)
    return text_chunks
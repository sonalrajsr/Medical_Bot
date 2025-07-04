# Medical Chat Bot

A Streamlit-based Medical Chat Bot that allows users to upload medical PDF documents, creates a searchable knowledge base, and answers user queries using state-of-the-art NLP models.

---

## Features

- **PDF Upload:** Upload any medical PDF document to build a custom knowledge base.
- **Document Chunking:** Automatically splits PDF content into manageable text chunks.
- **Vector Database:** Stores document embeddings for efficient semantic search using Pinecone.
- **Question Answering:** Uses a language model to answer user questions based on the uploaded document.
- **Interactive UI:** Simple and intuitive interface built with Streamlit.

---

## How It Works

1. **Upload PDF:** User uploads a medical PDF document.
2. **Processing:** The PDF is loaded, split into text chunks, and stored in a vector database.
3. **Ask Questions:** User enters a medical question related to the document.
4. **Retrieval:** The system retrieves relevant chunks from the vector database.
5. **LLM Response:** A language model generates an answer based on the retrieved context.

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/medical-bot.git
    cd medical-bot
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```sh
    python -m venv myenv
    myenv\Scripts\activate
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add your Pinecone API key and any other required keys:
      ```
      PINECONE_API_KEY=your_pinecone_api_key
      ```

---

## Usage

1. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

2. **In your browser:**
    - Upload a medical PDF.
    - Ask questions about the document in the provided input box.
    - View answers generated by the chatbot.

---

## Project Structure

```
medical-bot/
│
├── app.py                        # Streamlit app entry point
├── requirements.txt              # Python dependencies
├── Utility/
│   ├── LLM.py                    # Language model response logic
│   ├── get_data_from_pinecone.py # Pinecone retrieval logic
│   ├── pdf_chunk.py              # PDF chunking logic
│   ├── pdf_loader.py             # PDF loading logic
│   ├── pinecone_vector_db.py     # Pinecone vector DB logic
│   └── ...
├── Data/                         # (Optional) Directory for sample PDFs
└── ...
```

---

## Models Used

- **Question Answering:** [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2)
- **Embeddings:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **PDF Loading:** `langchain_community.document_loaders.PyPDFLoader`
- **Vector Store:** Pinecone

---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

---

## Notes

- This chatbot is for informational and educational purposes only. It is **not** a substitute for professional medical advice.
- Ensure you have a valid Pinecone API key for vector database functionality.

---

## License

MIT License

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LangChain](https://python.langchain.com/)
-
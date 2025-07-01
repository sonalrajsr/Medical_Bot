from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate

def get_llm_response(user_query: str, context: str) -> str:
    """
    Function to get a response from the LLM using a prompt template.
    Args:
        user_query (str): The user's question.
        context (str): The context in which the LLM should generate a response.
    Returns:
        str: The generated response from the LLM.
    """
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant related to medical topics. Context: {context}"),
        ("user", "{user_query}"),
    ])
    rendered_prompt = prompt_template.format(context=context, user_query=user_query)

    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa(question=user_query, context=rendered_prompt)
    return result['answer']
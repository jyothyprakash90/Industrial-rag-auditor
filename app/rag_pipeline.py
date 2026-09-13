from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from app import config

SYSTEM_PROMPT = """You are an industrial compliance auditor assistant.
Answer the user's question using ONLY the context provided below, which
was retrieved from an uploaded audit report.

Rules:
- If the answer is not contained in the context, say clearly that the
  report does not provide enough information to answer.
- Do not guess or use outside knowledge about compliance standards.
- When possible, mention which page(s) the answer came from, using the
  [PAGE X] tags in the context.
- Be concise and factual, as befits a compliance report.
"""


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=config.TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )


def retrieve_context(vectorstore: Chroma, question: str, k: int = None):
    """Return the top-k most relevant chunks for the question."""
    k = k or config.TOP_K
    return vectorstore.similarity_search(question, k=k)


def answer_question(vectorstore: Chroma, question: str) -> dict:
    """
    Full RAG call: retrieve relevant chunks, then generate a grounded
    answer using the LLM. Returns both the answer and the source
    chunks used, so the UI can display citations.
    """
    retrieved_docs = retrieve_context(vectorstore, question)
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": retrieved_docs,
    }

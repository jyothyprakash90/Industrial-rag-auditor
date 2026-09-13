from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

from app import config


def extract_text_from_pdf(file) -> str:
    """Extract raw text from an uploaded PDF file object."""
    reader = PdfReader(file)
    text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            # Tag each chunk of text with its page number so we can
            # cite sources later.
            text += f"\n[PAGE {page_num}]\n{page_text}"
    return text


def chunk_text(text: str) -> list[Document]:
    """Split raw text into overlapping chunks, preserving page tags."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    raw_chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in raw_chunks]


def build_vectorstore(documents: list[Document], collection_name: str = "audit_report") -> Chroma:
    """
    Embed document chunks and store them in a Chroma vector database.
    Each call creates a fresh in-memory collection scoped to the
    uploaded report, so different reports don't bleed into each other.
    """
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
    )
    return vectorstore


def ingest_pdf(file) -> Chroma:
    """End-to-end ingestion: PDF -> text -> chunks -> vectorstore."""
    text = extract_text_from_pdf(file)
    if not text.strip():
        raise ValueError(
            "No readable text found in this PDF. It may be scanned/image-only."
        )
    documents = chunk_text(text)
    vectorstore = build_vectorstore(documents)
    return vectorstore, len(documents)

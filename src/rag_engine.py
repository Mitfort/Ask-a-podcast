from typing import Any, TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class TranscriptSegment(TypedDict):
    text: str
    start: float | int
    end: float | int
    id: int | str


def create_vector_db(segments: list[TranscriptSegment], persist_directory: str = "db") -> Chroma:
    """
    Creates a vector database from the provided segments using HuggingFace embeddings and Chroma vector store.
    """

    texts: list[str] = []
    metadata: list[dict[str, str]] = []

    for seg in segments:
        texts.append(seg["text"])
        metadata.append({
            "start": str(seg["start"]),
            "end": str(seg["end"]),
            "id": str(seg["id"]),
        })

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = text_splitter.create_documents(texts=texts, metadatas=metadata)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)

    return vector_db
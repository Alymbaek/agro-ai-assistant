from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from pathlib import Path
from ..ml import predict
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]
VECTORSTORE_DIR = BASE_DIR / "rag" / "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_db = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

@tool
def ml_predict_tool(data: dict) -> dict:
    """
    Использует ML-модель для прогноза урожайности арбуза и оценки риска.
    Принимает погодные данные и регион.
    """
    y, risk = predict(data)
    return {"yield_index": y, "risk_level": risk}

@tool
def rag_recommend_tool(query: str) -> str:
    """
    Возвращает экспертные агрономические рекомендации из базы знаний
    по выращиванию арбуза (RAG).
    """
    docs = _db.similarity_search(query, k=3)
    return "\n".join(d.page_content for d in docs)







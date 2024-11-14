import os

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from embedding import get_openai_embedding


def get_retriever(source: str, embedding: Embeddings):
    return FAISS.load_local(
        f"./faiss/{source}",
        embedding,
        allow_dangerous_deserialization=True,
    ).as_retriever()

def get_vectorstore(source: str, embedding: Embeddings):
    return FAISS.load_local(
        f"./faiss/{source}",
        embedding,
        allow_dangerous_deserialization=True,
    )

def load_all_vectorstores(base_path: str, embedding: Embeddings):
    """Load all vectorstores in the specified base path."""
    vectorstores = {}
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"Loading vectorstore from {folder}")
            vectorstore = get_vectorstore(folder, embedding)
            vectorstores[folder] = vectorstore
    
    return vectorstores

if __name__ == "__main__":
    embedding = get_openai_embedding(api_key="")
    vectorstore = get_vectorstore("vectordata", embedding)
    relevant_documents = vectorstore.similarity_search_with_relevance_scores(
        "Trong trường hợp hai bên ký hợp đồng vay tài sản mà không có văn bản, chỉ bằng lời nói, nếu bên vay không trả tiền đúng hạn thì bên cho vay có quyền khởi kiện đòi nợ không?"
    )
    for doc in relevant_documents:
        print(doc)
        print()

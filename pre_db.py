from dotenv import load_dotenv
import os
from typing import List
from prepare_data import load_data
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from embedding import get_openai_embedding

def save_vector_db(
    documents: List[Document],
    embedding: Embeddings,
    path="./faiss/vectordata",
) -> FAISS:
    vector_store = FAISS.from_documents(
        documents=documents, 
        embedding=embedding
    )
    vector_store.save_local(path)
    return vector_store

if __name__ == "__main__":
    embedding = get_openai_embedding(api_key="")
    
    # Load all documents
    document = load_data("./datapickle/dan_su.pkl")
    document1 = load_data("./datapickle/dat_dai.pkl")
    document2 = load_data("./datapickle/hinh_su.pkl")
    document3 = load_data("./datapickle/hon_nhan_gia_dinh.pkl")
    document4 = load_data("./datapickle/lao_dong.pkl")  
    document5 = load_data("./datapickle/tai_chinh.pkl")
    
    # Combine all documents into one list
    all_documents = document + document1 + document2 + document3 + document4 + document5
    
    # Save all documents into a single vector store
    save_vector_db(all_documents, embedding, path="./faiss/vectordata")

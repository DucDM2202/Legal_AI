import os
from dotenv import load_dotenv
from pydantic import BaseModel
from chain import BasicRAG
from retriever import get_retriever
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
from embedding import get_openai_embedding
from langchain_openai import ChatOpenAI

load_dotenv()

embedding = get_openai_embedding(api_key="")
retrievers = get_retriever(source="dan_su", embedding=embedding)
# gg_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
gpt_llm = ChatOpenAI(temperature=0, openai_api_key="")
basicrag = BasicRAG(retrievers, gpt_llm)
print(basicrag.answer(question="Trong trường hợp hai bên ký hợp đồng vay tài sản mà không có văn bản, chỉ bằng lời nói, nếu bên vay không trả tiền đúng hạn thì bên cho vay có quyền khởi kiện đòi nợ không?"))
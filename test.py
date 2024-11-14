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
retrievers = get_retriever(source="vectordata", embedding=embedding)
# gg_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
gpt_llm = ChatOpenAI(temperature=0, openai_api_key="")
basicrag = BasicRAG(retrievers, gpt_llm)
print(basicrag.answer(question="Tôi cho bạn A vay với số tiền 10 triệu nhưng không kí hợp đồng và bạn A có hứa trả lại sau 3 tháng nhưng đến nay là 1 năm tôi chưa nhận được tiền, tôi có giục thì bạn nhất quyết không trả, tôi nên giải quyết như thế nào?"))
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

os.environ["HUGGIGNFACE_TOKEN"]=os.getenv["HUGGINGFACE_TOKEN"]
os.environ["LANGCHAIN_API_KEY"] = os.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple QnA Chatbot with OLlama"

embeddings=HuggingFaceEmbeddings(model="all-MiniLM-l6-v2")


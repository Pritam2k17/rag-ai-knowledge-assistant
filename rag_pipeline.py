import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

# Embeddings (choose one)
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# LLM (choose one)
from langchain.chat_models import ChatOpenAI

load_dotenv()

def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def create_rag_pipeline(file_path, use_openai=True):
    # Load document
    documents = load_document(file_path)

    # Split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Embeddings
    if use_openai and os.getenv("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    if use_openai and os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(temperature=0)
    else:
        return retriever  # fallback (no LLM)

    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa

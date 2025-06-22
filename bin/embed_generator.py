"""
Author: Nikhil Nageshwar Inturi (GitHub: @unikill066)
Date: 2025-06-22

Generate embeddings for PDF documents
"""

# imports
import os, warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# load environment variables
load_dotenv()

# logging - CHECK [x]

docs = list()
DOCS_DIR = "./docs"
PERSIST_DIR = "./chroma_db"

for _dir in [DOCS_DIR, PERSIST_DIR]:
    os.makedirs(_dir, exist_ok=True)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_paths = [os.path.join(DOCS_DIR, url) for url in os.listdir(DOCS_DIR) if url.endswith(".pdf")]  # loading the pdf files
# docs = [PyPDFLoader(url).load() for url in pdf_paths]
# docs_list = [item for sublist in docs for item in sublist]
for path in pdf_paths:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())  # flatten the doc list

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs)
# load and save the embeddings
vectorstore = Chroma.from_documents(documents=doc_splits, collection_name="rag-nik", embedding=embedding_model, persist_directory=PERSIST_DIR)
vectorstore.persist()
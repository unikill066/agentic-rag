"""
Author: Nikhil Nageshwar Inturi (GitHub: @unikill066)
Date: 2025-06-22

Retrieve documents from the vector store
"""

# imports
import os, warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# load environment variables
load_dotenv()

# logging - CHECK [x]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(persist_directory="/Users/discovery/Desktop/agentic-rag/chroma_db",  
    embedding_function=embedding_model,    
    collection_name="rag-nik")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
query = "Nikhil experience and education in length?"
# results = retriever.get_relevant_documents(query)
# for i, doc in enumerate(results, 1):
#     print(f"\n--- Doc #{i} ---\n{doc.page_content[:500]}…")

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

answer = qa({"query": query})
print("\n=== ANSWER ===\n", answer["result"])
print("\n=== SOURCES ===")
for src in answer["source_documents"]:
    print(" -", src.metadata.get("source", "unknown"))

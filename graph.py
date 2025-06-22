"""
Author: Nikhil Nageshwar Inturi (GitHub: @unikill066)
Date: 2025-06-22

Create a langgraph graph and compile it for invocation
"""

# imports
import warnings, os, logging, sys
from constants import COLLECTION_NAME
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import  HumanMessage
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
# load environment variables
load_dotenv()

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

class AgentState(TypedDict):  # agent state across the graph execution
    messages: Annotated[Sequence[BaseMessage], add_messages]

# creating a custom retriever tool for agentic tool use
# refer to bin/retriever.py
vectorstore = Chroma(persist_directory="/Users/discovery/Desktop/agentic-rag/chroma_db",  
    embedding_function=embedding_model,    
    collection_name=COLLECTION_NAME)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # k is the number of documents to retrieve
# vectorstore.as_retriever()
# query = ""
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
# retriever=retriever, return_source_documents=True)

retriever_tool = create_retriever_tool(
    retriever, 
    "retriever",
    """You are a specialized assistant and you have to search and return information about Nikhil from the documents
    Use the `retriever` tool **only** when the query explicitly related to Nikhil or queries about Nikhil.
    For all other queries, respond directly without using this custom `retriever` tool.
    And, for simple queries like 'hi', 'hello', or 'how are you', provide a short humanable response.
    """
)

tools = [retriever_tool, ]  # list of tools - Internet Search CHECK - [x]

# create a tool node
retriever_node = ToolNode([retriever_tool])


class router(BaseModel):
    route: str=Field(description="Route to 'yes' or 'no' based on relevance of query")

    
def rag_agent(state: AgentState) -> AgentState:
    logger.info("\n - - - RAG Agent Invocation - - -\n")
    messages = state["messages"]

    if len(messages) > 1:
        query = messages[-1].content
        RAG_AGENT_PROMPT = PromptTemplate(template = """
You are a helpful agent and you will answer any query:
Here is the query: {query}
""", input_variables=["query"])
        chain = RAG_AGENT_PROMPT | llm
        response = chain.invoke({"query": query})
        logger.info("\n - - - RAG Agent Response - - -\n")
        logger.info(response)
        return {"messages": [response]}
    else:
        llm_with_tools = llm.bind_tools(tools)  # add necessary web tools for advanced search
        response = llm_with_tools.invoke(messages)
        logger.info("\n - - - RAG Agent Response - - -\n")
        logger.info(response)
        return {"messages": [response]}

def document_quality(state: AgentState) -> Literal["rewrite", "generator"]:
    logger.info("\n - - - Document Quality Invocation - - -\n")
    messages = state["messages"]
    last_message = messages[-1]
    query = messages[0].content
    document = last_message.content
    llm_with_struct = llm.with_structured_output(router)
    prompt = PromptTemplate(template="""
    You are a helpful assistant and will check the quality and grade if a document is relevant to the query.
    Here is the query: {query}
    Here is the document: {context}
    If the document tabls about or contains information about the query, mark is as relevant.
    Return 'yes' or 'no' based on relevance.
    """, 
    input_variables=["context", "query"])
    chain = prompt | llm_with_struct
    response = chain.invoke({"context": document, "query": query})
    route_to = response.route

    if route_to == "yes":
        logger.info("Document is relevant to the query")
        return "generator"
    else:
        logger.info("Document is not relevant to the query")
        return "rewrite"

def generator(state: AgentState) -> AgentState:
    logger.info("\n - - - Generator Invocation - - -\n")
    messages = state["messages"]
    query = messages[0].content
    last_message = messages[-1]
    document = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm
    response = rag_chain.invoke({"context": document, "question": query})
    logger.info("\n - - - Generator Response - - -\n")
    logger.info(response)
    return {"messages": [response]}
    
def rewrite(state: AgentState) -> AgentState:
    logger.info("\n - - - Rewrite Invocation - - -\n")
    messages = state["messages"]
    query = messages[0].content
    message = [HumanMessage(
        content=f"""
        Below are the query and, digest it and try to reason about the underlying semantic meaning.
        Here is the initial query: {query}
        Come up with an improved query or an alternate query that is more relevant to the query.
        """,
    )]
    response = llm.invoke(message)
    logger.info("\n - - - Rewrite Response - - -\n")
    logger.info(response)
    return {"messages": [response]}

# # create a state graph
# graph = StateGraph(AgentState)
# graph.add_node("rag_agent", rag_agent)
# graph.add_node("retriever_node", retriever_node)
# graph.add_node("generator", generator)
# graph.add_node("rewrite", rewrite)
# graph.add_edge(START, "rag_agent")
# graph.add_conditional_edges("rag_agent", tools_condition, {"tools": "retriever_node", END: END})
# graph.add_conditional_edges("retriever_node", document_quality, {"generator": "generator", "rewrite": "rewrite"})
# graph.add_edge("rewrite", "rag_agent")
# graph.add_edge("generator", END)
# app = graph.compile()

def build_rag_state_graph():
    logger.info("\n - - - Building RAG State Graph - - -\n")
    graph = StateGraph(AgentState)  # stategraph definition
    # nodes
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("generator", generator)
    graph.add_node("rewrite", rewrite)
    # edges
    graph.add_edge(START, "rag_agent")
    graph.add_conditional_edges("rag_agent", tools_condition, {"tools": "retriever_node", END: END})
    graph.add_conditional_edges("retriever_node", document_quality, {"generator": "generator", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "rag_agent")
    graph.add_edge("generator", END)
    logger.info("\n - - - RAG State Graph Built - - -\n")
    return graph.compile()

# save compiled graph state to a PNG
def save_mermaid_graph(output_path: str = "./graph.png") -> None:
    """Generate the app’s Mermaid diagram and save it as a PNG file."""
    png_bytes = app.get_graph(xray=True).draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)

app = build_rag_state_graph()
save_mermaid_graph()
logger.info("\n - - - Graph saved to PNG - - -\n")
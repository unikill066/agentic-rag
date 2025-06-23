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
# load environment variablesx
load_dotenv()

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# validate openai api key
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables.")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, api_key=openai_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

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
    latest_message = messages[-1]
    query = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
    logger.info(f"Query received: {query}")
    # use tools for any query - let the LLM and tools_condition decide
    system_message = HumanMessage(content=f"""
    You are a helpful assistant that answers questions about Nikhil. 
    For ANY query about Nikhil (background, experience, education, projects, skills, work, etc.), 
    you MUST use the 'retriever' tool to search for relevant information first.
    For simple greetings like 'hi', 'hello', or 'how are you', respond directly without tools.
    Current query: {query}
    """)
    simple_greetings = ['hi', 'hello', 'hey', 'how are you', 'good morning', 'good afternoon', 'good evening']
    is_greeting = any(greeting.lower() in query.lower() for greeting in simple_greetings) and len(query.split()) <= 3
    if is_greeting:
        logger.info("Simple greeting - responding directly")
        response = llm.invoke([HumanMessage(content="Respond to this greeting in a friendly way: " + query)])
    else:
        logger.info("Using LLM with tools - letting tools_condition decide")
        llm_with_tools = llm.bind_tools(tools)
        enhanced_messages = [system_message] + messages
        response = llm_with_tools.invoke(enhanced_messages)
    logger.info(f"RAG Agent Response type: {type(response)}")
    logger.info(f"RAG Agent Response: {response}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Tool calls detected: {len(response.tool_calls)} tool(s)")
        for i, tool_call in enumerate(response.tool_calls):
            logger.info(f"Tool call {i+1}: {tool_call}")
    else:
        logger.info("No tool calls in response")
    return {"messages": [response]}

def document_quality(state: AgentState) -> Literal["rewrite", "generator"]:
    logger.info("\n - - - Document Quality Invocation - - -\n")
    messages = state["messages"]
    
    if len(messages) < 2:
        logger.info("Not enough messages for quality check - going to rewrite")
        return "rewrite"

    original_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    
    if not original_query:
        logger.info("No original query found - going to rewrite")
        return "rewrite"
    
    last_message = messages[-1]
    document = last_message.content if hasattr(last_message, 'content') else str(last_message)
    logger.info(f"Checking quality for query: {original_query}")
    logger.info(f"Document snippet: {document[:200]}...")
    llm_with_struct = llm.with_structured_output(router)
    prompt = PromptTemplate(template="""
    You are a helpful assistant checking document relevance.
    Query: {query}
    Document: {context}
    Is this document relevant to answering the query? 
    - If the document contains information that can help answer the query, return 'yes'
    - If the document is not relevant or doesn't contain useful information, return 'no'
    """, input_variables=["context", "query"])
    chain = prompt | llm_with_struct
    response = chain.invoke({"context": document, "query": original_query})
    route_to = response.route.lower()
    logger.info(f"Quality check result: {route_to}")
    if route_to == "yes":
        logger.info("Document is relevant - going to generator")
        return "generator"
    else:
        logger.info("Document is not relevant - going to rewrite")
        return "rewrite"

def generator(state: AgentState) -> AgentState:
    logger.info("\n - - - Generator Invocation - - -\n")
    messages = state["messages"]
    original_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    last_message = messages[-1]
    document = last_message.content if hasattr(last_message, 'content') else str(last_message)
    logger.info(f"Generating answer for: {original_query}")
    try:
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm
        response = rag_chain.invoke({"context": document, "question": original_query})
    except Exception as e:
        logger.error(f"Error with hub prompt: {e}")
        # Fallback prompt
        fallback_prompt = PromptTemplate(template="""
        Based on the following context, answer the question:
        Context: {context}
        Question: {question}
        Answer:""", input_variables=["context", "question"])
        rag_chain = fallback_prompt | llm
        response = rag_chain.invoke({"context": document, "question": original_query})
    logger.info(f"Generator Response: {response}")
    return {"messages": [response]}
    
def rewrite(state: AgentState) -> AgentState:
    logger.info("\n - - - Rewrite Invocation - - -\n")
    messages = state["messages"]
    original_query = None
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    if not original_query:
        original_query = "Tell me about Nikhil"
    logger.info(f"Rewriting query: {original_query}")
    rewrite_prompt = PromptTemplate(template="""
    The original query was: {query}
    The retrieval didn't find relevant information. Please rewrite this query to be more specific and likely to find relevant information about Nikhil's background, experience, or qualifications.
    Rewritten query:""", input_variables=["query"])
    chain = rewrite_prompt | llm
    response = chain.invoke({"query": original_query})
    logger.info(f"Rewritten query: {response}")
    rewritten_message = HumanMessage(content=response.content if hasattr(response, 'content') else str(response))
    return {"messages": [rewritten_message]}

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
def save_mermaid_graph(app, output_path: str = "./graph.png") -> None:
    """Generate the app’s Mermaid diagram and save it as a PNG file."""
    png_bytes = app.get_graph(xray=True).draw_mermaid_png()
    with open(output_path, "wb") as f:
        f.write(png_bytes)

app = build_rag_state_graph()
save_mermaid_graph(app)
logger.info("\n - - - Graph saved to PNG - - -\n")
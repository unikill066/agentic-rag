"""
Author: Nikhil Nageshwar Inturi (GitHub: @unikill066)
Date: 2025-06-22

Create a streamlit app for the Agentic RAG chatbot
"""

# imports
import streamlit as st, os, sys, logging
try:
    from langchain_core.messages import HumanMessage
except:
    from langchain.schema import HumanMessage, BaseMessage
from graph import app
# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# validate openai api key
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables.")

@st.cache_resource
def init_connection():
    """
    Initialize connection to Firebase Firestore using the firebase credentials
    """
    pass

st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="centered")
st.title("Agentic RAG Chatbot")

st.caption("Powered by LangGraph and OpenAI models")

try:
    st.image("misc/mtr1.png")
    st.caption("This image is generated using FLUX image generator - Black Forest Labs: [Repo](https://github.com/unikill066/FLUX-image-generator/tree/main) | [Live Demo](https://flux-image-generator-58dpp94unlwkzkaz5hmagy.streamlit.app/)")
except:
    st.caption("Image not found - continuing without image")

with st.expander("Note:"):
    st.write("""This chatbot, powered by the GPT‑3.5‑turbo language model, is designed to answer questions about Nikhil's professional background, publications, projects, and qualifications. Conversations are stored to help improve the quality of responses. Please keep inquiries respectful and avoid personal or inappropriate topics.""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.messages:
    with st.chat_message("assistant"):
        welcome_message = """
        Welcome! I'm **Resume Bot**, a virtual assistant designed to provide insights into Nikhil's background and qualifications. 

        Feel free to inquire about any aspect of Nikhil's profile, such as:

        - His Master's in Machine Learning and AI from Purdue Global
        - His experience at Infosys and Aganitha Cognitive Solutions
        - His proficiency in programming languages and ML/Generative AI frameworks
        - His experience in drug discovery and development
        - His passion for leveraging technologies to drive innovation

        What would you like to know first? I'm ready to answer your questions in detail.
        """
        st.markdown(welcome_message)

if prompt := st.chat_input("Ask me about Nikhil's background, projects, publications, etc..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                human_message = HumanMessage(content=prompt)
                st.write("🔍 Processing your query...")
                logger.info(f"User query: {prompt}")
                
                # graph invocation
                response = app.invoke({"messages": [human_message]})
                if response and "messages" in response and len(response["messages"]) > 0:
                    last_message = response["messages"][-1]
                    if hasattr(last_message, 'content'):
                        assistant_response = last_message.content
                    else:
                        assistant_response = str(last_message)
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    
                    logger.info(f"Response generated successfully")
                else:
                    error_msg = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logger.error(f"Error processing query: {e}")

# debug-info section
with st.expander("Debug Info"):
    st.write("Session State Messages:", st.session_state.messages)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
import streamlit as st
from langchain_core.messages import HumanMessage
from graph import build_rag_state_graph, AgentState



@st.cache_resource
def init_connection():
    """
    Initialize connection to Firebase Firestore using the firebase credentials
    """
    cred = credentials.Certificate(firebase_credentials)
    firebase_admin.initialize_app(cred)
    return firestore.client()


st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="centered")
st.title("Agentic RAG Chatbot")


st.caption("Powered by LangGraph and OpenAI models")
st.image("misc/mtr1.png")
st.caption("This image is generated using FLUX image generator - Black Forest Labs: [Repo](https://github.com/unikill066/FLUX-image-generator/tree/main) | [Demo](https://flux-image-generator-58dpp94unlwkzkaz5hmagy.streamlit.app/)")
with st.expander("Note:"):
    st.write("""This chatbot, powered by the GPT‑3.5‑turbo language model, is designed to answer questions about Nikhil’s professional background, publications, projects, and qualifications. Conversations are stored to help improve the quality of responses. Please keep inquiries respectful and avoid personal or inappropriate topics.""")




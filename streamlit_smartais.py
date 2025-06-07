# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ------------- 1) Load environment variables -------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    st.error("🔑 Please set your OPENAI_API_KEY in the .env file.")
    st.stop()

# ------------- 2) Sidebar settings & file loader -------------
#st.sidebar.title("⚙️ Settings")
source_type = st.sidebar.radio("Select document type:", ("Plain Text (.txt)", "PDF (.pdf)"))

if source_type == "Plain Text (.txt)":
    data_path = st.sidebar.text_input("Path to your .txt file:", "/Users/vishalkagade/Downloads/smartals_instructions_english.txt")
    loader = TextLoader(data_path, encoding="utf-8")
else:
    data_path = st.sidebar.text_input("Path to your .pdf file:", "")
    loader = PyPDFLoader(data_path)

# ------------- 3) Build or load FAISS index -------------
@st.cache_resource
def load_or_build_vectorstore(data_path, _loader, _embeddings):
    docs = _loader.load()
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=2000,
        chunk_overlap=200,
    )
    texts, metadatas = [], []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            meta = {**doc.metadata, "chunk_index": i}
            metadatas.append(meta)
    vect = FAISS.from_texts(texts, embedding=_embeddings, metadatas=metadatas)
    return vect

@st.cache_resource(show_spinner=False)
def get_conversational_chain(_vectorstore):
    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=False,
    )

# Instantiate embeddings once
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# Load or build vectorstore (cached)
with st.spinner("📚 Loading documents and building vector store…"):
    try:
        vectorstore = load_or_build_vectorstore(data_path, loader, embeddings)
    except Exception as e:
        st.error(f"❌ Error loading your document: {e}")
        st.stop()

# Create conversational chain (cached)
retrieval_chain = get_conversational_chain(vectorstore)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI
st.title("SmartAIS Onboardinfg Assistant")
st.markdown("Ask questions about the onboarding process.")

# Show chat history
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Vishal_the_bot:** {bot_msg}")
    st.markdown("---")

# Input form with automatic clearing
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", key="input_box")
    submit = st.form_submit_button("Send")

if submit and user_input:
    with st.spinner("🤖 Thinking..."):
        try:
            result = retrieval_chain({"question": user_input, "chat_history": st.session_state.chat_history})
            answer = result["answer"].strip()
            st.write(f"Answer received: {answer}")  # debug print in UI
        except Exception as e:
            st.error(f"Error during response generation: {e}")
            answer = None

    if answer:
        st.session_state.chat_history.append((user_input, answer))


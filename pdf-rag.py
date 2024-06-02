"""
This script sets up a Streamlit application for handling PDF and webpage data ingestion, vector storage, and querying 
using a language model from HuggingFace. The application allows users to upload PDF files or enter URLs, process the 
content, and ask questions about the content using a retrieval-augmented generation approach.

Modules:
    - streamlit: For creating the web interface.
    - llama_index.core: For handling the core operations of the vector store and index.
    - llama_index.llms.huggingface: For HuggingFace language model inference.
    - llama_index.embeddings.huggingface: For HuggingFace embeddings.
    - dotenv: For loading environment variables.
    - os: For handling directory and file operations.
    - base64: For encoding PDF files.
    - re: For regular expressions to handle strings.
    - shutil: For file operations like deleting directories.

Functions:
    - get_hf_token: Retrieves the HuggingFace token from environment variables.
    - configure_llm: Configures the language model and embedding settings.
    - create_vector_store_directories: Creates directories for persistent storage and data.
    - build_vector_store: Processes the uploaded file or URL and builds the vector store.
    - displayPDF: Displays a PDF file in the Streamlit app.
    - remove_special_characters: Removes special characters from a string.
    - remove_special_characters_from_url: Removes special characters from the last part of a URL.
    - create_vector_store: Creates a vector store from documents and persists it.
    - data_ingestion_pdf: Loads PDF documents from a directory and creates a vector store.
    - data_ingestion_web: Loads documents from a URL and creates a vector store.
    - delete_vector_store: Deletes the loaded vector store.
    - handle_query: Handles user queries by interacting with the vector store and returning responses.
"""

import streamlit as st
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    download_loader,
    VectorStoreIndex,
    SimpleDirectoryReader,
    ChatPromptTemplate,
    Settings,
)
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
import os
import base64
import re
import shutil


def get_hf_token():
    """
    Retrieves the HuggingFace token from environment variables.

    Returns:
        str: The HuggingFace token.
    """
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        token = st.secrets["HF_TOKEN"]
    return token


def configure_llm(token):
    """
    Configures the language model and embedding settings.

    Args:
        token (str): The HuggingFace token.
    """
    Settings.llm = HuggingFaceInferenceAPI(
        model_name="google/gemma-1.1-7b-it",
        tokenizer_name="google/gemma-1.1-7b-it",
        context_window=3000,
        token=token,
        max_new_tokens=2048,
        generate_kwargs={"temperature": 0.0},
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def create_vector_store_directories():
    """
    Creates directories for persistent storage and data.

    Returns:
        tuple: Paths for persistent storage and data directories.
    """
    PERSIST_DIR = "./db"
    DATA_DIR = "./data"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return PERSIST_DIR, DATA_DIR


def build_vector_store(uploaded_file, url):
    """
    Processes the uploaded file or URL and builds the vector store.

    Args:
        uploaded_file (UploadedFile): The uploaded PDF file.
        url (str): The URL to process.
    """
    with st.spinner("Processing..."):
        if url:
            vector_store_directory = PERSIST_DIR + "/" + remove_special_characters_from_url(url)
            data_ingestion_web(url, vector_store_directory)
        elif uploaded_file:
            data_directory = DATA_DIR + "/" + remove_special_characters(uploaded_file.name)
            os.makedirs(data_directory, exist_ok=True)
            filepath = data_directory + "/" + uploaded_file.name
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            displayPDF(filepath)
            vector_store_directory = PERSIST_DIR + "/" + remove_special_characters(uploaded_file.name)
            data_ingestion_pdf(data_directory, vector_store_directory)
        st.success("Done")


def displayPDF(file):
    """
    Displays a PDF file in the Streamlit app.

    Args:
        file (str): The file path of the PDF to display.
    """
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def remove_special_characters(s):
    """
    Removes special characters from a string.

    Args:
        s (str): The input string.

    Returns:
        str: The cleaned string with special characters removed.
    """
    pattern = r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", s)


def remove_special_characters_from_url(url):
    """
    Removes special characters from the last part of a URL.

    Args:
        url (str): The input URL.

    Returns:
        str: The cleaned last part of the URL.
    """
    parts = url.split("/")
    last_part = parts[-1]
    pattern = r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", last_part)


def create_vector_store(documents, persist_dir):
    """
    Creates a vector store from documents and persists it.

    Args:
        documents (list): A list of document objects.
        persist_dir (str): The directory where the vector store will be persisted.
    """
    l_index = VectorStoreIndex.from_documents(documents)
    os.makedirs(persist_dir, exist_ok=True)
    l_index.storage_context.persist(persist_dir=persist_dir)
    st.session_state.vector_store_directory = persist_dir


def data_ingestion_pdf(data_dir, persist_dir):
    """
    Loads PDF documents from a directory and creates a vector store.

    Args:
        data_dir (str): The directory containing PDF documents.
        persist_dir (str): The directory where the vector store will be persisted.
    """
    documents = SimpleDirectoryReader(data_dir).load_data()
    create_vector_store(documents, persist_dir)


def data_ingestion_web(url, persist_dir):
    """
    Loads documents from a URL and creates a vector store.

    Args:
        url (str): The URL to load documents from.
        persist_dir (str): The directory where the vector store will be persisted.
    """
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=[url])
    create_vector_store(documents, persist_dir)


def delete_vector_store(i_selected_store):
    """
    Deletes the loaded vector store.

    Args:
        i_selected_store (str): The name of the selected vector store to delete.
    """
    l_store_directory = PERSIST_DIR + "/" + i_selected_store
    shutil.rmtree(l_store_directory)
    l_data_directory = DATA_DIR + "/" + i_selected_store
    try:
        shutil.rmtree(l_data_directory)
    except FileNotFoundError:
        l_data_directory = ''
    st.info("Loaded vector store " + i_selected_store + " deleted.")


def handle_query(query, persist_dir):
    """
    Handles user queries by interacting with the vector store and returning responses.

    Args:
        query (str): The user query.
        persist_dir (str): The directory where the vector store is persisted.

    Returns:
        str: The response to the user query.
    """
    l_storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    l_index = load_index_from_storage(l_storage_context)
    chat_text_qa_msgs = [
        (
            "user",
            """You are a Q&A assistant.
               Your main goal is to provide answers as accurately as possible,
               based on the instructions and context you have been given.
               If a question does not match the provided context or is outside the scope of the document,
               kindly advise the user to ask questions within the context of the document.
            Context:
            {context_str}
            Question:
            {query_str}
            """,
        )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    query_engine = l_index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)

    if hasattr(answer, "response"):
        return answer.response
    elif isinstance(answer, dict) and "response" in answer:
        return answer["response"]
    else:
        return "Sorry, I couldn't find an answer."


# Main script execution
token = get_hf_token()
configure_llm(token)
PERSIST_DIR, DATA_DIR = create_vector_store_directories()

# Streamlit app initialization
st.title("(PDF and Webpage) Information and Inference🗞️")
st.markdown("Retrieval-Augmented Generation")
st.markdown("Start chat ...🚀")

# Get list of existing vector stores
existing_vector_stores = [d for d in os.listdir(PERSIST_DIR) if os.path.isdir(os.path.join(PERSIST_DIR, d))]

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Upload a PDF or specify a URL and ask me anything about its content.",
        }
    ]

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files or enter a URL and Click on the Submit & Process Button")
    url = st.text_input("Enter URL")
    if st.button("Submit & Process"):
        build_vector_store(uploaded_file, url)

    selected_store = st.selectbox("Select Vector Store", [""] + existing_vector_stores)
    side_left, side_right = st.sidebar.columns(2)
    if selected_store and side_left.button("Delete Loaded Vector Store"):
        delete_vector_store(selected_store)
    if selected_store and side_right.button("Delete Chat Messages"):
        st.session_state.messages = []

user_prompt = st.chat_input("Ask me anything about the content of the PDF or the Webpage:")
if user_prompt:
    if selected_store:
        vector_store_directory = PERSIST_DIR + "/" + selected_store
    if not vector_store_directory:
        vector_store_directory = st.session_state.vector_store_directory

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    response = handle_query(user_prompt, vector_store_directory)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

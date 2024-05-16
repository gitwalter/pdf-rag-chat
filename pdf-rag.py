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

# Load environment variables
load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    token = st.secrets["HF_TOKEN"]

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token=token,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.0},
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "./data"

vector_store_directory = ""

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)


def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def remove_special_characters(s):
    # Define a regular expression pattern to match special characters
    pattern = r"[^a-zA-Z0-9\s]"
    # Use the sub() function to replace all occurrences of the pattern with an empty string
    return re.sub(pattern, "", s)


def remove_special_characters_from_url(url):
    # Split the URL by "/"
    parts = url.split("/")
    # Get the last part of the URL
    last_part = parts[-1]
    # Define a regular expression pattern to match special characters
    pattern = r"[^a-zA-Z0-9\s]"
    # Use the sub() function to replace all occurrences of the pattern with an empty string
    cleaned_last_part = re.sub(pattern, "", last_part)
    return cleaned_last_part


def create_vector_store(documents, persist_dir):
    l_index = VectorStoreIndex.from_documents(documents)
    os.makedirs(persist_dir, exist_ok=True)
    l_index.storage_context.persist(persist_dir=persist_dir)
    st.session_state.vector_store_directory = persist_dir


def data_ingestion_pdf(data_dir, persist_dir):
    documents = SimpleDirectoryReader(data_dir).load_data()
    create_vector_store(documents, persist_dir)


def data_ingestion_web(url, persist_dir):
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=[url])
    create_vector_store(documents, persist_dir)


def display_delete_button(i_selected_store):
    # Button to delete the loaded vector store
    if st.button("Delete Loaded Vector Store"):
        l_store_directory = PERSIST_DIR + "/" + i_selected_store
        shutil.rmtree(l_store_directory)
        l_data_directory = DATA_DIR + "/" + i_selected_store
        try:
            shutil.rmtree(l_data_directory)
        except FileNotFoundError:
            l_data_directory = ''
            # for webpages no data-file exists
            
        st.info("Loaded vector store " + i_selected_store + " deleted.")


def handle_query(query, persist_dir):
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


# Streamlit app initialization
st.title("(PDF and Webpage) Information and Inference🗞️")
st.markdown("Retrieval-Augmented Generation")
st.markdown("start chat ...🚀")

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
    uploaded_file = st.file_uploader(
        "Upload your PDF Files or enter a URL and Click on the Submit & Process Button"
    )

    url = st.text_input("Enter URL")

    # Dropdown to select pre-existing vector stores
    selected_store = st.selectbox("Select Vector Store", [""] + existing_vector_stores)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if url:
                vector_store_directory = (
                    PERSIST_DIR + "/" + remove_special_characters_from_url(url)
                )
                data_ingestion_web(url, vector_store_directory)

            elif uploaded_file:
               
                data_directory = (
                    DATA_DIR + "/" + remove_special_characters(uploaded_file.name)
                )
                
                os.makedirs(data_directory, exist_ok=True)
                
                filepath = (
                    data_directory
                    + "/"
                    + uploaded_file.name
                )                
                
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                displayPDF(filepath)

                vector_store_directory = (
                    PERSIST_DIR + "/" + remove_special_characters(uploaded_file.name)
                )                

                data_ingestion_pdf(data_directory, vector_store_directory)

            st.success("Done")

if selected_store:
    display_delete_button(selected_store)

user_prompt = st.chat_input(
    "Ask me anything about the content of the PDF or the Webpage:"
)
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

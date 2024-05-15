import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, download_loader,\
     VectorStoreIndex, SimpleDirectoryReader, ChatPromptTemplate, Settings
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

import os
import base64


# Load environment variables
load_dotenv()

token=os.getenv("HF_TOKEN")

if not token:
    token = st.secrets["HF_TOKEN"]

# Configure the Llama index settings
Settings.llm = HuggingFaceInferenceAPI(
    model_name="google/gemma-1.1-7b-it",
    tokenizer_name="google/gemma-1.1-7b-it",
    context_window=3000,
    token = token,
    max_new_tokens=1024,
    generate_kwargs={"temperature": 0.0},
)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Define the directory for persistent storage and data
PERSIST_DIR = "./db"
DATA_DIR = "data"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def data_ingestion_pdf():    
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def data_ingestion_web(url):      
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    loader = BeautifulSoupWebReader()
    documents = loader.load_data(urls=[url])    
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

def handle_query(query):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
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
        """
    )
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
    
    query_engine = index.as_query_engine(text_qa_template=text_qa_template)
    answer = query_engine.query(query)
    
    if hasattr(answer, 'response'):
        return answer.response
    elif isinstance(answer, dict) and 'response' in answer:
        return answer['response']
    else:
        return "Sorry, I couldn't find an answer."


# Streamlit app initialization
st.title("(PDF and Webpage) Information and Inference🗞️")
st.markdown("Retrieval-Augmented Generation") 
st.markdown("start chat ...🚀")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF or specify a URL and ask me anything about its content.'}]

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files or enter a URL and Click on the Submit & Process Button")
    
    url = st.text_input("Enter URL")
    
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if url:
                data_ingestion_web(url)
            else:
                filepath = "data/saved_pdf.pdf"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                displayPDF(filepath)  # Display the uploaded PDF
                data_ingestion_pdf()  # Process PDF every time new file is uploaded
            st.success("Done")

user_prompt = st.chat_input("Ask me anything about the content of the PDF or the Webpage:")
if user_prompt:
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = handle_query(user_prompt)
    st.session_state.messages.append({'role': 'assistant', "content": response})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])
# PDF and Webpage Information and Inference App

Welcome to the PDF and Webpage Information and Inference App! This application allows you to upload PDF files or enter URLs of webpages, process their content, and perform retrieval-augmented generation for information extraction and Q&A. Built with Streamlit and LlamaIndex, it leverages HuggingFace models for embedding and inference.

## Features

- **Upload PDFs:** Upload and process PDF files for content analysis.
- **Webpage Processing:** Enter URLs to ingest and analyze webpage content.
- **Vector Store Creation:** Create and persist vector stores for efficient retrieval.
- **Query Interface:** Ask questions about the processed content and get accurate answers.
- **Interactive Chat:** Engage in a chat interface for querying the content.
- **Persistent Storage:** Save and load vector stores for re-use.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/gitwalter/pdf-rag-chat.git
    cd pdf-rag-chat
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
   Create a `.env` file in the project root and add your HuggingFace API token and email:
    ```env
    HF_TOKEN=your_huggingface_api_token
    HF_MAIL=your_email_address
    ```

4. **Run the Streamlit app:**
    ```sh
    streamlit run pdf-rag.py
    ```

## Usage

1. **Upload PDFs or Enter URLs:**
   - Use the sidebar to upload a PDF file or enter a URL.
   - Click the "Submit & Process" button to process the content.

2. **Select Vector Store:**
   - Choose from pre-existing vector stores in the dropdown menu.

3. **Ask Questions:**
   - Use the chat input at the bottom to ask questions about the processed content.
   - The app will provide accurate answers based on the content.

4. **Delete Vector Store:**
   - Use the delete button to remove a selected vector store if no longer needed.

## File Structure

├── pdf-rag.py # Main application file

├── requirements.txt # Required Python packages

├── .env # Environment variables (not included in the repo)

├── db/ # Directory for persistent vector stores

├── data/ # Directory for uploaded PDFs and webpage data

└── README.md # This README file #

## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing the framework for the web app.
- [LlamaIndex](https://github.com/jerryjliu/llama) for the indexing and retrieval functionalities.
- [HuggingFace](https://huggingface.co/) for the inference and embedding models.

Enjoy using the PDF and Webpage Information and Inference App! 🚀

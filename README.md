PDF and Webpage Information and Inference App
Welcome to the PDF and Webpage Information and Inference App! This application allows you to upload PDF files or enter URLs of webpages, process their content, and perform retrieval-augmented generation for information extraction and Q&A. Built with Streamlit and LlamaIndex, it leverages HuggingFace models for embedding and inference.

Features
Upload PDFs: Upload and process PDF files for content analysis.
Webpage Processing: Enter URLs to ingest and analyze webpage content.
Vector Store Creation: Create and persist vector stores for efficient retrieval.
Query Interface: Ask questions about the processed content and get accurate answers.
Interactive Chat: Engage in a chat interface for querying the content.
Persistent Storage: Save and load vector stores for re-use.
Installation
Clone the repository:

sh
Code kopieren
git clone https://github.com/yourusername/pdf-webpage-inference-app.git
cd pdf-webpage-inference-app
Install dependencies:

sh
Code kopieren
pip install -r requirements.txt
Set up environment variables:
Create a .env file in the project root and add your HuggingFace API token:

env
Code kopieren
HF_TOKEN=your_huggingface_api_token
Run the Streamlit app:

sh
Code kopieren
streamlit run app.py
Usage
Upload PDFs or Enter URLs:

Use the sidebar to upload a PDF file or enter a URL.
Click the "Submit & Process" button to process the content.
Select Vector Store:

Choose from pre-existing vector stores in the dropdown menu.
Ask Questions:

Use the chat input at the bottom to ask questions about the processed content.
The app will provide accurate answers based on the content.
Delete Vector Store:

Use the delete button to remove a selected vector store if no longer needed.
File Structure
bash
Code kopieren
.
├── app.py                   # Main application file
├── requirements.txt         # Required Python packages
├── .env                     # Environment variables (not included in the repo)
├── db/                      # Directory for persistent vector stores
├── data/                    # Directory for uploaded PDFs and webpage data
└── README.md                # This README file
Contributing
We welcome contributions to improve this project! If you have any ideas, suggestions, or issues, please create a pull request or open an issue on GitHub.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Streamlit for providing the framework for the web app.
LlamaIndex for the indexing and retrieval functionalities.
HuggingFace for the inference and embedding models.
Enjoy using the PDF and Webpage Information and Inference App! 🚀

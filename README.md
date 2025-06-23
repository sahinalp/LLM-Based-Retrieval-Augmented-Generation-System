# LLM-Based Retrieval-Augmented Generation (RAG) System

This project contains a chatbot system that utilizes Retrieval-Augmented Generation (RAG) to answer user queries based on PDF documents. The system retrieves relevant information from documents using vector similarity search and generates contextual answers using the LLaMA-7B language model.

The goal of this project is to develop a question-answering chatbot that provides accurate, context-aware responses by combining large language models with external document data.


## 🔧 Tools & Technologies

* [x] **Python:** Core programming language used to build the backend and manage document processing, embeddings, and model integration.
* [x] **Flask:** Lightweight web framework used to build the chatbot interface.
* [x] **Hugging Face Transformers:** Used to load and run the LLaMA-7B language model for response generation.
* [x] **ChromaDB:** Vector database used for storing and retrieving document embeddings based on similarity search.
* [x] **Sentence-Transformers:** Library used to convert text chunks into semantic embeddings.
* [x] **Poppler:** PDF rendering tool used for extracting raw text from PDF files.
* [x] **NLTK:** Used for sentence-level tokenization during the chunking process.
* [x] **Torch (PyTorch):** Deep learning framework used as the backend for the LLM (LLaMA-7B).
* [x] **HTML/CSS/JavaScript:** Used to design and style the chatbot interface on the frontend.

## 🗂️ Project Structure

```
├── app.py                  # Main Flask app that initializes the chatbot interface
├── requirements.txt        # Python dependencies for the virtual environment
├── dataset/                # Folder containing source PDF documents
├── chroma_db/              # ChromaDB vector store (auto-generated on first run)
├── Helpers/                # Custom helper modules for RAG pipeline
│   ├── pdfHelper.py        # Extracts and chunks text from PDFs
│   ├── embeddingHelper.py  # Generates and stores embeddings using Sentence-Transformers
│   └── modelHelper.py      # Handles prompt construction and LLM answer generation
├── templates/
│   └── index.html          # Frontend HTML structure for the chatbot
├── static/
│   └── style.css           # Styling (CSS) for the web interface
└── README.md               # Project documentation
```

## 🚀 Installation & Setup Instructions

1. **Install Poppler**  
   Download and extract [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/download/v24.08.0-0/Release-24.08.0-0.zip) to the root folder of this project.

2. **Create Virtual Environment**
   ```console
   python3.12 -m venv .venv
   ```
   **Note:** If you don’t have virtualenv installed, first run:
   ```console
   pip install virtualenv
   ```

3. **Activate Virtual Environment**
   ```console
   .\.venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```console
   pip install -r requirements.txt
   ```
   Optional (for GPU acceleration with CUDA 12.8):
   ```console
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
   visit [pytorch.org](https://pytorch.org/get-started/locally/) for more configurations.

5. **Run the Application**
   ```console
   python app.py
   ```

6. **Access the Chat Interface**
   Open your browser and go to: http://127.0.0.1:5000/

7. **Ask Questions**
   Enter your question in the message input at the bottom and press **Enter** or click **Send**. A loading spinner will appear during response generation.

###   **Notes**
* On the first run, the system will extract text from PDFs and create a ChromaDB index. This may take some time.
* The loading spinner next to the send button indicates that the model is generating a response.

## ⚙️ System Architecture
1. **PDF Processing**

   Handled by [pdfHelper.py](https://github.com/sahinalp/LLM-Based-Retrieval-Augmented-Generation-System/blob/main/Helpers/pdfHelper.py)

   Extracts raw text and splits it into sentences

2. **Embedding & Storage**  

   Handled by [embeddingHelper.py](https://github.com/sahinalp/LLM-Based-Retrieval-Augmented-Generation-System/blob/main/Helpers/embeddingHelper.py)

   Computes sentence embeddings and stores them in a ChromaDB collection
    
   On query, the top-N most relevant chunks (e.g., top 5) are retrieved using similarity search

3. **Retrieval & Generation:**
  
   Handled by [modelHelper.py](https://github.com/sahinalp/LLM-Based-Retrieval-Augmented-Generation-System/blob/main/Helpers/modelHelper.py)

   Retrieved chunks are passed as prompt context to LLaMA-7B

   The model generates an answer based on both the retrieved content and the question

## 💬 Example Questions & Answers

❓ **Question:**
What is the maximum percentage of face fiber wear allowed under the Lifetime warranty?

✅ **Answer:**
Milliken warrants that the carpet will lose no more than ten percent (10%) of its face fiber by weight during the Lifetime of the carpet. If the carpet is installed on stairs this warranty will be limited to five years.

❓ **Question:**
Which cleaning system does Milliken recommend for deep carpet cleaning and why?

✅ **Answer:**
Milliken recommends the MilliCare Dry Carpet Cleaning system. MilliCare Textile and Carpet Care® is Green Seal Certified and an IICRC Certified training provider.

**Example Image:**

![image](https://github.com/user-attachments/assets/e3e6dd01-9a7c-4533-8fc5-822dfe0a29bd)


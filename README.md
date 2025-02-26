# Nepali PDF Chatbot

## Overview

The **Nepali PDF Chatbot** is a Streamlit-based application that allows users to:
- Upload a PDF (written in Nepali).
- Ask questions in **English** and receive responses in **English or Nepali**.
- Extract text from the PDF and process it into embeddings using **FAISS** and **HuggingFace**.
- Leverage **Google's Gemini AI model** for intelligent responses.

---

## Features

- 📂 **Upload Nepali PDF** – Extract text from a PDF document.
- 🔍 **Semantic Search** – Uses FAISS for efficient similarity-based retrieval.
- 🔤 **Multilingual Support** – Accepts queries in **English** and provides responses in **English or Nepali**.
- ⚡ **Fast Response** – Utilizes **Gemini AI** for generating responses.
- 🛠️ **Robust Text Processing** – Handles text cleaning, chunking, and embedding generation.

---

## Technologies Used

The chatbot utilizes the following technologies:

1. **Python**
2. **Streamlit** – For UI
3. **PyPDF2** – For PDF text extraction
4. **LangChain** – For text chunking and retrieval
5. **FAISS** – For vector storage
6. **HuggingFace Embeddings** – For text similarity search
7. **Deep Translator** – For language translation using **Google Translator API**
8. **Google Gemini API** – For AI-generated responses

---

## Installation

### Prerequisites

Ensure you have the following installed:

- **Python 3.8+**
- **Pip**
- **Virtual environment** (optional but recommended)

### Setup

#### Clone the Repository

-   git clone https://github.com/BibekSapkota1/Nepali-Pdf-chatbot.git
-   cd nepali-pdf-chatbot
#### Create and Activate a Virtual Environment

 -  python -m venv venv  # Create virtual environment
 -  source venv/bin/activate  # Activate on macOS/Linux
 -  venv\Scripts\activate  # Activate on Windows

#### Install Dependencies

 -  pip install -r requirements.txt

#### Set Up Environment Variables

Create a .env file in the project directory and add:

   GEMINI_API_KEY=your_gemini_api_key_here

#### Usage

Run the Streamlit App

   - streamlit run chatbot.py

- Upload a Nepali PDF in the sidebar.

- Ask Questions about the document.

- View Responses in English or Nepali.

### File Structure

📂 nepali-pdf-chatbot //
│-- 📄 chatbot.py             # Main Streamlit app //
│-- 📄 requirements.txt   # List of dependencies //
│-- 📄 .env               # API key configuration //

### Troubleshooting

If the chatbot does not initialize, ensure your GEMINI_API_KEY is correctly set in .env.

If translation fails, Google Translator API might be overloaded. Retry after some time.

### Future Improvements

Add support for multiple PDF uploads.

Implement more advanced NLP models for better question-answering.

Optimize embeddings for faster search performance.


###### Author (Bibek Sapkota)


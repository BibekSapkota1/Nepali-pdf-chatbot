## Nepali PDF Chatbot

Overview

The Nepali PDF Chatbot is a Streamlit-based application that allows users to upload a PDF (written in Nepali), ask questions in English, and receive responses in either English or Nepali. The system extracts text from the PDF, processes it into embeddings using FAISS and HuggingFace, and leverages Google's Gemini AI model for intelligent responses.

Features

📂 Upload Nepali PDF: Extract text from a PDF document.

🔍 Semantic Search: Uses FAISS for efficient similarity-based retrieval.

🔤 Multilingual Support: Accepts queries in English and provides responses in either English or Nepali.

⚡ Fast Response: Utilizes Gemini AI for generating responses.

🛠️ Robust Text Processing: Handles text cleaning, chunking, and embedding generation.

Technologies Used

Python

Streamlit (for UI)

PyPDF2 (for PDF text extraction)

LangChain (for text chunking and retrieval)

FAISS (for vector storage)

HuggingFace Embeddings (for text similarity search)

Deep Translator (for language translation)

Google Gemini API (for AI-generated responses)

Installation

Prerequisites

Python 3.8+

Pip

Virtual environment (optional but recommended)

Setup

Clone the Repository

   git clone https://github.com/BibekSapkota1/Nepali-PDF-chatbot.git
   cd nepali-pdf-chatbot

Create and Activate a Virtual Environment

   python -m venv venv  # Create virtual environment
   source venv/bin/activate  # Activate on macOS/Linux
   venv\Scripts\activate  # Activate on Windows

Install Dependencies

   pip install -r requirements.txt

Set Up Environment Variables
Create a .env file in the project directory and add:

   GEMINI_API_KEY=your_gemini_api_key_here

Usage

Run the Streamlit App

   streamlit run chatbot.py

Upload a Nepali PDF in the sidebar.

Ask Questions about the document.

View Responses in English or Nepali.

File Structure

📂 nepali-pdf-chatbot
│-- 📄 chatbot.py             # Main Streamlit app
│-- 📄 requirements.txt   # List of dependencies
│-- 📄 .env               # API key configuration

Troubleshooting

If the chatbot does not initialize, ensure your GEMINI_API_KEY is correctly set in .env.

If translation fails, Google Translator API might be overloaded. Retry after some time.

Future Improvements

Add support for multiple PDF uploads.

Implement more advanced NLP models for better question-answering.

Optimize embeddings for faster search performance.


Author
Bibek Sapkota




import os
os.environ["PYDANTIC_ALLOW_REUSE_VALIDATOR"] = "1"
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Nepali PDF Chatbot",
    page_icon=":books:",
    layout="wide"
)

import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import google.generativeai as genai
import time
from typing import List, Dict



# Load environment variables and configure API
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Please set GEMINI_API_KEY in your .env file")
    st.stop()

# Initialize Gemini
@st.cache_resource

def init_gemini():
    """Initialize Gemini API"""
    try:
        genai.configure(api_key=gemini_api_key)
        # Test if the API is working by creating a model instance
        model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')
        return model  # Return the model instance
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return None
            

def extract_text_from_pdf(pdf_file):
    """Extract and clean text from PDF."""
    try:
        reader = PdfReader(pdf_file)
        text = ""
        
        st.write(f"Processing {len(reader.pages)} pages from PDF...")
        
        for page in reader.pages:
            # Extract and clean text
            page_text = page.extract_text()
            # Convert to UTF-8
            page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
            text += page_text + "\n\n"
        
        # Clean the text
        text = ' '.join(text.split())
        
        # Show sample
    #   st.write("Sample of extracted text:", text[:200])
        
        return text
        
    except Exception as e:
        st.error(f"PDF extraction error: {str(e)}")
        return None


def chunk_text(text: str, chunk_size: int = 5000, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "।", ".", " "]  # Including Nepali sentence separator
        )
        chunks = splitter.split_text(text)
        return [chunk.encode('utf-8', errors='ignore').decode('utf-8') for chunk in chunks]
    except Exception as e:
        st.error(f"Error chunking text: {str(e)}")
        return None

def create_embeddings(chunks: List[str]):
    """Create embeddings using HuggingFace model."""
    try:
        # Use a lighter model
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None
 

#--------------------------------------------------------------------------------
# This is right code
def safe_translate(text: str, dest_lang: str, retries: int = 3) -> str:
    """Safely translate text with retries."""
    for attempt in range(retries):
        try:
            translator = GoogleTranslator(source='auto', target=dest_lang)
            translated = translator.translate(text)
            if dest_lang == 'ne':
                return translated.encode('utf-8', errors='ignore').decode('utf-8')
            return translated
        except Exception as e:
            if attempt == retries - 1:
                st.warning(f"Translation failed: {str(e)}. Using original text.")
                return text
            time.sleep(1)
#--------------------------------------------------------------------------------

def setup_chat_engine(vectorstore):
    """Setup the chat engine with Gemini."""
    try:
        model = init_gemini()
        if model is None:
            return None
            
        chat = model.start_chat(history=[])
        
        def process_query(user_query: str, response_language: str = "en") -> str:
            try:
                # Get more relevant chunks and increase the similarity search
                relevant_chunks = vectorstore.similarity_search(user_query, k=5)
                
                # Clean and translate chunks
                translated_chunks = []
                for chunk in relevant_chunks:
                    clean_text = chunk.page_content.encode('utf-8', errors='ignore').decode('utf-8')
                    clean_text = ' '.join(clean_text.split())
                    translated_chunks.append(clean_text)  # Keep original text first
                
                if not translated_chunks:
                    return "I couldn't find any relevant information in the document."
                
                context = "\n".join(translated_chunks)
                # prompt = f"""You are an AI assistant analyzing a document. The document appears to be: {context}

                # Please provide a detailed answer to the following question: {user_query}

                # If the question is about the general content of the document, summarize what you can see in the text.
                # If the question is about a specific topic not present in the document, clearly state that the information is not present.
                # """
                prompt = f"""You are an AI assistant analyzing a Nepali document. Here is the relevant context:

                Context: {context}

                Question: {user_query}

                Instructions:
                - Provide a clear and detailed answer based on the context above
                - If the information is not in the context, explicitly state that
                - Keep the response concise and focused on the question
                - If the context is unclear or contains mixed languages, mention this in your response
                """
                
                response = chat.send_message(prompt)
                
                if response_language == "ne":
                    try:
                        return GoogleTranslator(source='en', target='ne').translate(response.text)
                    except Exception:
                        return response.text
                return response.text
                
            except Exception as e:
                st.error(f"Error in query processing: {str(e)}")
                return "Sorry, I couldn't process your question. Please try again."
        
        return process_query
            
    except Exception as e:
        st.error(f"Error in chat engine setup: {str(e)}")
        return None
#--------------------------------------------------------------------------------

def create_ui():
    """Create the Streamlit UI."""
    st.title("🇳🇵 Nepali PDF Chatbot")
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📚 Document Upload")
        uploaded_file = st.file_uploader("Upload a Nepali PDF", type="pdf")
        
        st.header("🔤 Language Settings")
        response_language = st.radio("Response Language:", ("English", "Nepali"))
        
        if uploaded_file:
            st.success("PDF uploaded successfully!")
    
    # Main chat interface
    if uploaded_file:
        # Initialize or get session state
        if 'processed_pdf' not in st.session_state:
            st.session_state.processed_pdf = False
            st.session_state.chat_history = []
        
        # Process PDF if not already done
        if not st.session_state.processed_pdf:
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    chunks = chunk_text(text)
                    if chunks:
                        vectorstore = create_embeddings(chunks)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.processed_pdf = True
                            st.success("✅ PDF processed successfully!")
        
        # Chat interface
        if st.session_state.processed_pdf:
            # Add error handling for chat engine setup
            try:
                process_query = setup_chat_engine(st.session_state.vectorstore)
                if process_query is None:
                    st.error("Failed to initialize chat engine. Please try again.")
                    return
                
                # Chat input
                user_question = st.chat_input("Ask a question about the PDF...")
                if user_question:
                    # Add user message to chat
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    
                    # Get and add assistant response
                    with st.spinner("Thinking..."):
                        try:
                            response = process_query(user_question, "ne" if response_language == "Nepali" else "en")
                            if response:
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                            else:
                                st.error("Failed to get response from the chat engine.")
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                
                # Display chat history
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        st.write("You:", message["content"])
                    else:
                        if response_language == "Nepali":
                            st.markdown(
                                f"""<div style='font-family: "Arial Unicode MS", "Noto Sans", sans-serif;'>
                                🤖 Bot: {message["content"]}</div>""", 
                                unsafe_allow_html=True
                            )
                        else:
                            st.write("🤖 Bot:", message["content"])
            except Exception as e:
                st.error(f"Error setting up chat interface: {str(e)}")
    
    else:
        # Welcome message when no PDF is uploaded
        st.info("👋 Welcome! Please upload a Nepali PDF document to start chatting.")
        
def main():
    # Set up page configuration
  #  st.set_page_config(page_title="Nepali PDF Chatbot", page_icon="🇳🇵", layout="wide")
    
    # Add custom CSS for Nepali font support
    st.markdown("""
        <style>
        .nepali-text {
            font-family: "Arial Unicode MS", "Noto Sans", sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize Gemini
    if not init_gemini():
        st.error("Failed to initialize the chatbot. Please check your API key.")
        return
    
    # Create the UI
    create_ui()

if __name__ == "__main__":
    main()
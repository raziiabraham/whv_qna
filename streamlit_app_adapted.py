import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from pathlib import Path

# Load the OpenAI API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_response(openai_api_key, query_text):
    # Prepares the system prompt for Bahasa Indonesia responses
    system_prompt = "Always generate output in Bahasa Indonesia!\n"
    query_text = system_prompt + query_text  # Appending the actual query text to the system prompt

    # Path to the document
    doc_path = Path(__file__).parent / "scraped_data.txt"
    
    # Read the document
    with open(doc_path, 'r', encoding='utf-8') as file:
        documents = [file.read()]
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    texts = text_splitter.create_documents(documents)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    
    # Create retriever interface
    retriever = db.as_retriever()
    
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    
    return qa.run(query_text)

st.set_page_config(page_title='🇮🇩 Working Holiday Visa 101 🇦🇺')
st.title('🇮🇩 Working Holiday Visa 101 🇦🇺')

query_text = st.text_input('Question / Pertanyaan:', placeholder='Please provide your question here / Masukkan pertanyaan Anda di sini.')

# Process the query
if query_text:
    with st.spinner('Sedang mencari jawaban...'):
        response = generate_response(openai_api_key, query_text)
    st.info(response if response else "Mohon maaf saya tidak bisa menjawab peternyaan tersebut, mohon menanykan hal yang lain.")
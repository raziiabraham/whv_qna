__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import streamlit as st
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pathlib import Path

# Access the OpenAI API key from secrets.toml
openai_api_key = st.secrets["openai"]["api_key"]

# Define the function to generate response using RAG
def generate_response(openai_api_key, query_text):
    # Prepares the system prompt for Bahasa Indonesia responses
    system_prompt = "Always generate output in Bahasa Indonesia! Output must be no more than 100 words!\n"
    query_text = system_prompt + query_text  # Appending the actual query text to the system prompt

    # Path to the document
    doc_path = Path(__file__).parent / "scraped_data.txt"
    
    # Read the document
    with open(doc_path, 'r', encoding='utf-8') as file:
        documents = [file.read()]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
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

# Setup the page title to be displayed on Streamlit
st.set_page_config(page_title='🇮🇩 Working Holiday Visa 101 🇦🇺')
st.title('🇮🇩 Working Holiday Visa 101 🇦🇺')

# Setup the input textfield to take questions from user
query_text = st.text_input('Question / Pertanyaan:', placeholder='Please provide your question here / Masukkan pertanyaan Anda di sini.')

# Process the query input by users
if query_text:
    with st.spinner('Sedang mencari jawaban...'):
        response = generate_response(openai_api_key, query_text)
    st.info(response if response else "Mohon maaf saya tidak bisa menjawab pertanyaan tersebut, mohon menanyakan hal yang lain.")

# Display sources from where the answers will be generated
st.write('\n\n\n') # Add more line spaces
st.markdown("**The answers are generated from the following sources as of April 25, 2024. / Jawaban dihasilkan dari sumber-sumber berikut, per tanggal 25 April 2024.**")
st.markdown("* https://immi.homeaffairs.gov.au/what-we-do/whm-program/")
st.markdown("* https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/work-holiday-462")
st.markdown("* https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/work-holiday-462/first-work-holiday-462")
st.markdown("* https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/work-holiday-462/second-work-holiday-462")
st.markdown("* https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/work-holiday-462/third-work-and-holiday-462")
st.markdown("* https://indonesia.embassy.gov.au/jakt/visa462.html")
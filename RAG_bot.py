#pip install langchain 
#pip install chromadb 
#pip install -U langchain-huggingface
#pip install streamlit


# Function to install a package using pip
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the required packages
#install_package("langchain")
#install_package("chromadb")
#install_package("langchain-huggingface
#install_package("streamlit")
#install_package("langchain-openai")

# Your main code goes here
# Example: A print statement after installation
#print("All packages installed, continuing with the script...")

from langchain import hub
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain.document_loaders import TextLoader #for .txt files
from pathlib import Path #to load more than one document
from dotenv import load_dotenv
#%pip install chromadb
#%pip install -U langchain-huggingface
#pip install streamlit
import bs4
import subprocess
import sys
import getpass
import os
import chromadb
import subprocess
import sys
import streamlit as st


# Directly pass the API key to the ChatCohere class
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

llm = ChatCohere(model="command-r", cohere_api_key=cohere_api_key)


# SPECIFY THE FOLDER PATH WHERE ALL YOUR .TXT FILES ARE LOCATED!!
folder_path = "documents"  # Replace with your folder path

# Use pathlib to list all .txt files in the folder
file_paths = list(Path(folder_path).rglob("*.txt"))

# Load documents from each .txt file in the folder
all_documents = []
for file_path in file_paths:
    loader = TextLoader(str(file_path), encoding='utf-8')  # Convert Path object to string
    documents = loader.load()
    all_documents.extend(documents)  # Add documents to the list

#from langchain_community.document_loaders import PyPDFLoader #for PDFs
#loader = PyPDFLoader("documents/Liquid_NN.pdf")
#documents_pdf = loader.load()

    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_documents)

### EMBEDDINGS
embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

chroma_client = chromadb.Client()

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
{"context": retriever | format_docs, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

#system_prompt = """
#You are an immigration expert for Austria. Answer the user's questions about immigration policies, processes, and requirements with accurate and helpful information.
#"""

# Define the function to run the RAG pipeline with the system prompt
#def run_rag_pipeline_with_system_prompt(question):
    # Combine system prompt, retrieved documents, and user question
#    system_context = system_prompt + "\n\n"
    
#    # Combine the system prompt and the user's question
#    formatted_question = system_context + f"Question: {question}"
    
#    # Invoke the RAG pipeline with the formatted input
#    return rag_chain.invoke(formatted_question)


# Define the function to run the RAG pipeline
def run_rag_pipeline(question):
    #return rag_chain.invoke(question)
    st.info(rag_chain.invoke(question))

# Streamlit interface
st.title("Immigration Bot")

# User input field
#user_input = st.text_input("Hello there. I will assist you with questions regarding your immigration to Austria.")

# If the user submits a question, run the RAG pipeline and display the response
#if user_input:
#    response = run_rag_pipeline(user_input)
#    st.write(f"Chatbot: {response}")


with st.form("my_form"):
    text = st.text_area(
        "Hello there. I will assist you with questions regarding your immigration to Austria.",
        "Enter your question here",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        run_rag_pipeline(text)



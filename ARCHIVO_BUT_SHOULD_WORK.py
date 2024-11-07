import cohere
import chromadb
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere import CohereEmbeddings
from pathlib import Path
from langchain.schema import Document

# Initialize Cohere client with your API key
cohere_api_key = "GVO78KFjnCgLq4jv7jvRd6SnuDTXbUOr03l4rcUo"
co = cohere.Client(cohere_api_key)

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Step 1: Load documents (example with text files)
def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
    return documents

# Step 2: Use Cohere to generate embeddings
embedding_model = "embed-multilingual-light-v3.0"


folder_path = "documents"

def create_documents_from_strings(strings):
    documents = [Document(page_content=text) for text in strings]
    return documents

documents = load_documents(file_paths)
documents = create_documents_from_strings(documents)
document_chunks = text_splitter.split_documents(documents)


# Load your documents
file_paths = list(Path(folder_path).rglob("*.txt"))  # Add your file paths here
documents = load_documents(file_paths)

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
document_chunks = text_splitter.split_documents(documents)

# Step 3: Store document chunks in ChromaDB
vectorstore = Chroma.from_documents(documents=document_chunks, embedding=embedding_model)

# Step 4: Set up the retriever to retrieve relevant documents
retriever = vectorstore.as_retriever()

# Step 5: Define the system prompt
system_prompt = """
You are an AI assistant that provides helpful, concise answers based on retrieved documents.
"""

# Step 6: RAG pipeline - Retrieval and generation
def rag_pipeline(user_input):
    # Retrieve relevant documents
    relevant_docs = retriever.get_relevant_documents(user_input)

    # Format the retrieved documents into a string
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Create a final prompt including the system prompt, context, and user input
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {user_input}\n\nAI:"

    # Step 7: Use Cohere to generate a response
    response = co.generate(
        model='command-r-plus-08-2024',  # Replace with a valid Cohere model ID
        prompt=full_prompt,
        max_tokens=200,
        temperature=0.6  # Adjust temperature to control creativity
    )

    # Return the generated response
    return response.generations[0].text

# Example usage
user_question = "What are the benefits of a plant-based diet?"
response = rag_pipeline(user_question)

# Print the response
print(f"AI Response: {response}")
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_documents(directory):
    """Load all .txt files from a directory."""
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            file_path = os.path.join(directory, file)
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def main():
    # Load documents from the "data" directory
    docs = load_documents("data")
    
    # Split documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)
    
    # Initialize embeddings with an explicit model to avoid deprecation warnings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create a FAISS index from the document chunks and save it locally
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    
    print("Documents have been indexed and saved to 'faiss_index'.")

if __name__ == "__main__":
    main()


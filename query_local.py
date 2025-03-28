import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def main():
    # Initialize embeddings with an explicit model name
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # Load the persisted FAISS vector store with deserialization enabled
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print("Error while loading FAISS index:", e)
        return
    
    # Set up a retriever; adjust k (number of documents) as needed
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Initialize ChatOpenAI (GPT-4) using the updated import
    llm = ChatOpenAI(model="gpt-4", openai_api_key="")
    
    # Create a RetrievalQA chain (using the "stuff" method to combine docs)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    query = input("Enter your question: ")
    answer = qa_chain.invoke(query)
    print("Answer:", answer)

if __name__ == "__main__":
    main()


import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#
st.set_page_config(page_title="c++ RAG chatbot")
st.title("C++ RAG chatbot")
st.write("Ask any question realted to c++ Introduction")

@st.cache_resource
def load_vectorstore():
    loadder = TextLoader('C++_Introduction.txt', encoding="UTF-8")
    documents = loadder.load()
    
    textSplitter =RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    finalDocuments=textSplitter.split_documents(documents)
    
    
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")
    
    
    db=FAISS.from_documents(finalDocuments,embeddings)
    
    return db
    
db=load_vectorstore()

query=st.text_input("Enter your question for c++")
if query:
    docs=db.similarity_search(query,k=3)
    st.subheader("Retrived Context")
    for i , doc in enumerate(docs):
        st.markdown(f"**Result{i+1}:**")
        st.write(doc.page_content)
        
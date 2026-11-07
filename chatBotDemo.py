import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="c++ RAG chatbot")
st.title("C++ RAG chatbot")
st.write("Ask any question realted to c++ Introduction")

@st.cache_resource
def load_vectorstore():
    loadder = TextLoader(r'C:\Users\ASUS\Downloads\ChatBot-main\ChatBot-main\C++_Introduction.txt', encoding="UTF-8")
    documents = loadder.load()
    
    textSplitter =RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n\n---\n\n", "\n\n", "\n", " ", ""])
    finalDocuments=textSplitter.split_documents(documents)
    
    
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")
    
    
    db=FAISS.from_documents(finalDocuments,embeddings)
    
    return db
    
db=load_vectorstore()

query=st.text_input("Enter your question for c++")

prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question.
If the answer is not in the context, say: "I don't know based on the provided document."

Context:
{context}

Question:
{question}
""")

# ollama LLM model
llm = OllamaLLM(model="gemma2:2b")
output_parser = StrOutputParser()

# LCEL chain
chain = (
    {
        "context": lambda x: "\n\n".join(doc.page_content for doc in x["docs"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | llm
    | output_parser
)

if query:
    # docs=db.similarity_search(query,k=3)
    # for i , docs in enumerate(docs):
    #     st.subheader(f"Context {i+1}")
    #     st.write(docs.page_content)
    # st.write("---------------------------------------------------")
    
    docs2=db.similarity_search(query,k=3)
    response = chain.invoke({"docs" : docs2,"question": query}) 
    st.subheader("Retrived Context")
    st.write(response)
        

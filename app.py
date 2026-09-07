from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")

#title
st.title("StudyBuddy")

st.markdown("I'm Ur AI study Assistant")


#user query
query=st.text_input(label="enter text")


#prompt template
prompt=ChatPromptTemplate(
    [
        ("system","act as student assistant ,answer all the query from the user"),
        ("user","context:{context} \n query:{query}")
    ]
)


#llm
llm=ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


#output parser
output_parser=StrOutputParser()


#LCEL chaining
chain=prompt|llm|output_parser


#file loading
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])



if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("uploaded.pdf")
    texts = loader.load()


    #text splitting
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=splitter.split_documents(texts)


    #embeddings model
    embedding=OllamaEmbeddings(model="nomic-embed-text")


    #vectorstore db
    db=Chroma.from_documents(chunks,embedding)


    #similarity search
    context_text=db.similarity_search(query)
    context="\n".join(page.page_content for page in context_text)

    
    #final resul invoking llm
    res=chain.invoke({"context":context,"query":query})

    #final output
    st.success(res)
else:
    #handling for normal llm without context,normally gives answers based on query
    res=chain.invoke({"context":"","query":query})
    st.success(res)

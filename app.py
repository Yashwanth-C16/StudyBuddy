from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")


#embediing fetch once store for future use
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )


#title
st.title("StudyBuddy")

st.markdown("I'm Ur AI study Assistant")


#user query
query=st.text_input(label="enter text")


#prompt template
prompt=ChatPromptTemplate(
    [
        ("system","Act as a student assistant. Answer the user's question using the provided context. If the context contains the answer, use it. Do not use outside knowledge when answering questions about the PDF."),
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

#button
button=st.button("query")

if button:
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        loader = PyMuPDFLoader(tmp_path)
        texts = loader.load()


        #text splitting
        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        chunks=splitter.split_documents(texts)


        #embeddings model
        embedding = get_embedding_model()


        #vectorstore db
        db=Chroma.from_documents(chunks,embedding)


        #similarity search
        context_text=db.similarity_search(query)
        context="\n".join(page.page_content for page in context_text)

        st.write(context)
        #final resul invoking llm
        res=chain.invoke({"context":context,"query":query})

        #final output
        st.success(res)
    else:
        #handling for normal llm without context,normally gives answers based on query
        res=chain.invoke({"context":"","query":query})
        st.success(res)

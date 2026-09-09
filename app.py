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

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


# embedding model - fetch once, cache for future use
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=gemini_api_key
    )


# title
st.title("StudyBuddy")
st.markdown("I'm Ur AI study Assistant")


# user query
query = st.text_input(label="enter text")


# prompt template
prompt = ChatPromptTemplate(
    [
        ("system", "Act as a student assistant. Answer the user's question using the provided context. If the context contains the answer, use it. Do not use outside knowledge when answering questions about the PDF."),
        ("user", "context:{context} \n query:{query}")
    ]
)


# llm
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)


# output parser
output_parser = StrOutputParser()


# LCEL chaining
chain = prompt | llm | output_parser


# file loading
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# button
button = st.button("query")

if button:
    # 1. empty query check
    if not query.strip():
        st.warning("Please enter a question.")

    elif uploaded_file:
        # 2. build/rebuild vector DB only if it's a new file
        if "db" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                loader = PyMuPDFLoader(tmp_path)
                texts = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(texts)

                if not chunks:
                    st.error("Couldn't extract any text from this PDF. It may be scanned or image-based.")
                    st.stop()

                embedding = get_embedding_model()
                st.session_state.db = Chroma.from_documents(chunks, embedding)
                st.session_state.file_name = uploaded_file.name

            except Exception as e:
                st.error(f"Failed to process the PDF: {e}")
                st.stop()

        # 3. retrieval + answer generation
        try:
            context_text = st.session_state.db.similarity_search(query)
            context = "\n\n".join(page.page_content for page in context_text)

            res = chain.invoke({"context": context, "query": query})
            st.success(res)

            with st.expander("📄 Sources used for this answer"):
                for i, doc in enumerate(context_text, start=1):
                    page_num = doc.metadata.get("page", "unknown")
                    if isinstance(page_num, int):
                        page_num += 1  # PyMuPDF pages are 0-indexed
                    st.markdown(f"**Source {i} — Page {page_num}**")
                    st.caption(doc.page_content[:300] + "...")

        except Exception as e:
            st.error(f"Something went wrong getting a response: {e}")

    else:
        # no PDF uploaded - plain LLM fallback
        try:
            res = chain.invoke({"context": "", "query": query})
            st.success(res)
        except Exception as e:
            st.error(f"Something went wrong getting a response: {e}")
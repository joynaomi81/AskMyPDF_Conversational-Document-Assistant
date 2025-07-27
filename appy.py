import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Gemini API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return [Document(page_content=text)]

# Streamlit config
st.set_page_config(page_title="AskMyPDF", layout="wide")
st.title("📄 AskMyPDF")

# Upload file
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    docs = extract_text_from_pdf(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Embedding and Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)

    # User question input
    with st.form(key="qa_form"):
        question = st.text_input("Ask a question about the PDF:")
        submitted = st.form_submit_button("Submit")

    if submitted and question:
        docs_similar = db.similarity_search(question)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs_similar, question=question)

        # Save Q&A to session state
        st.session_state.chat_history.append((question, answer))

    # Display Q&A history
    for q, a in st.session_state.chat_history:
        st.markdown("**Question:** " + q)
        st.markdown("**Answer:** " + a)
        st.markdown("---")

    # Optional: Clear history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

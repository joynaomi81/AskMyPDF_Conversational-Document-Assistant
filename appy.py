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

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return [Document(page_content=text)]

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("📄 AskMyPDF")

# Initialize session state to store chat history and DB
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = None

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and st.session_state.db is None:
    docs = extract_text_from_pdf(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create Vector DB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.db = FAISS.from_documents(chunks, embeddings)

# Question input box
question = st.text_input("Ask a question about the PDF:")

# Generate answer when a question is asked
if question and st.session_state.db:
    docs_similar = st.session_state.db.similarity_search(question)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    chain = load_qa_chain(llm, chain_type="stuff")
    result = chain.run(input_documents=docs_similar, question=question)

    # Save Q&A to history
    st.session_state.chat_history.append((question, result))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Chat History")
    for q, a in reversed(st.session_state.chat_history):  # reverse to show latest on top
        st.markdown(f"**🧑‍💻 You:** {q}")
        st.markdown(f"**🤖 Gemini:** {a}")
        st.markdown("---")

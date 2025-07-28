import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import tempfile
import speech_recognition as sr

# ✅ Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ✅ Function: Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()
    return [Document(page_content=text)]

# ✅ Function: Extract text from audio
def extract_text_from_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, couldn't understand the audio."
    except sr.RequestError:
        return "API unavailable."

# ✅ Set Streamlit config
st.set_page_config(page_title="PDF Q&A", layout="wide")

# ✅ Title and Menu Bar
st.title("📄 AskMyPDF – Conversational Document Assistant")

st.sidebar.header("📊 Progress Tracker")
progress_bar = st.sidebar.progress(0)
progress_label = st.sidebar.empty()

# ✅ Initialize state
if "db" not in st.session_state:
    st.session_state.db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "progress" not in st.session_state:
    st.session_state.progress = 0

# ✅ Input Tabs: PDF | Text | Audio
tabs = st.tabs(["📄 Upload PDF", "📋 Paste Text", "🎙️ Upload Audio"])

with tabs[0]:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        docs = extract_text_from_pdf(uploaded_file)
        source = "PDF"

with tabs[1]:
    text_input = st.text_area("Paste your document text here:")
    if text_input.strip():
        docs = [Document(page_content=text_input)]
        source = "Text"

with tabs[2]:
    audio_file = st.file_uploader("Upload an audio file (WAV format)", type=["wav"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        extracted_text = extract_text_from_audio(tmp_path)
        docs = [Document(page_content=extracted_text)]
        source = "Audio"

# ✅ Chunk and embed
if 'docs' in locals():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    st.session_state.text_chunks = chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)
    st.session_state.db = db
    st.success(f"{source} uploaded and processed successfully.")

# ✅ Progress Meter (based on chat history)
if st.session_state.text_chunks:
    total_chunks = len(st.session_state.text_chunks)
    read_chunks = len(st.session_state.chat_history)
    progress = min(read_chunks / total_chunks, 1.0)
    progress_bar.progress(progress)
    progress_label.write(f"{int(progress * 100)}% of document queried.")

# ✅ Q&A Form
if st.session_state.db:
    with st.form("question_form"):
        question = st.text_input("Ask me any question about the material:")
        submitted = st.form_submit_button("Submit")

    if submitted and question:
        docs_similar = st.session_state.db.similarity_search(question)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        chain = load_qa_chain(llm, chain_type="stuff")
        result = chain.run(input_documents=docs_similar, question=question)
        st.session_state.chat_history.append((question, result))

# ✅ Chat History
if st.session_state.chat_history:
    st.markdown("## 💬 Chat History")
    for i, (q, a) in enumerate(reversed(st.session_state.chat_history), 1):
        st.markdown(f"**Q{i}:** {q}")
        st.markdown(f"**A{i}:** {a}")

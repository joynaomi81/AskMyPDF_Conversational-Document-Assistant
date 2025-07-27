import os
import streamlit as st
import tempfile

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# ✅ Load API key from secrets.toml
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ✅ Initialize Gemini model and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ✅ Streamlit UI setup
st.set_page_config(page_title="📘 Conversational Tutor (Gemini)", layout="wide")
st.title("📘 Conversational Tutor AI (Gemini 1.5 Flash)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ File uploaded successfully!")

    # Load & split
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    # Vectorstore
    db = FAISS.from_documents(chunks, embedding_model)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    st.markdown("---")
    st.subheader("🧠 Ask Any Question")
    question = st.text_input("Enter your question:")

    if question:
        result = qa_chain.run(question)
        st.markdown(f"**Answer:** {result}")

    st.markdown("---")
    st.subheader("📝 Generate Quiz Questions")
    if st.button("Generate Questions"):
        prompt = f"""
        You are a tutor. Based on the content below, generate 3 comprehension questions.

        Content:
        {chunks[0].page_content[:2000]}
        """
        response = llm.invoke(prompt)
        st.markdown(response.content)

    st.markdown("---")
    st.subheader("🎯 Grade My Answer")
    input_question = st.text_input("Question to grade:")
    input_answer = st.text_area("Your answer:")

    if st.button("Grade My Answer"):
        grading_prompt = f"""
        Grade the student's answer based on the question and the original content.

        Question: {input_question}
        Student's Answer: {input_answer}
        Content: {chunks[0].page_content[:2000]}

        Provide a score over 10 and a short explanation.
        """
        grade_result = llm.invoke(grading_prompt)
        st.markdown(grade_result.content)

    os.remove(tmp_path)

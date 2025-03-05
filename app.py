import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

# Function to split text into chunks
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Function to generate embeddings and save to FAISS index
def get_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chains():
    prompt_template = """
    Answer the question in detail as much as possible from the provided context. Make sure to provide all details.
    If the answer is not present in the context, just say "Answer not present in the context." Do not provide any other information.
    
    Context: {context}\n
    Question: {question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user query
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chains()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

# Main function for Streamlit appimport shutil

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("PDF Chatbot with Gemini AI")
    st.write("Upload PDF documents and ask questions!")

    # Upload PDF files
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if pdf_docs:
        with st.spinner("Processing PDFs..."):
            # **Delete previous FAISS index**
            if os.path.exists("faiss_index"):
                shutil.rmtree("faiss_index")  # Remove old index

            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_chunks(raw_text)
            get_embeddings(text_chunks)  # Create new FAISS index
        
        st.success("PDFs processed successfully!")

    # User query input
    user_question = st.text_input("Ask a question from the uploaded PDFs:")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Searching for the answer..."):
                response = user_input(user_question)
            st.write("### Reply:")
            st.write(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
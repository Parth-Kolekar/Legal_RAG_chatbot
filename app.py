import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the path for the FAISS index
FAISS_INDEX_PATH = "faiss_index"

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

# FIX 1: Removed the unused 'vector_store' argument from the function definition.
def get_conversational_chain():
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", dont provide the wrong answer.\n\n
    Context: \n{context}\nQuestion: \n{question}\n
    
    Answer:
    
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS index
    new_db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Perform similarity search
    docs = new_db.similarity_search(user_question)
    
    # Get the conversational chain and run it
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Legal RAG Chatbot", page_icon=":robot:")
    st.title("Legal RAG Chatbot using Gemini Pro")
    
    # Sidebar for PDF uploads
    with st.sidebar:
        st.title("Upload PDF Documents")
        pdf_docs = st.file_uploader("Upload PDF files and click on submit", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDF documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF documents processed and vector store created successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    st.header("Ask a Question")

    # FIX 2: Restructured the Q&A section to use a button.
    # This prevents API calls on every keystroke.
    if os.path.exists(FAISS_INDEX_PATH):
        user_question = st.text_input("Ask a question about the legal documents:", key="user_question")
        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Finding the answer..."):
                    user_input(user_question)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please upload and process PDF documents in the sidebar to begin.")


if __name__ == "__main__":
    main()
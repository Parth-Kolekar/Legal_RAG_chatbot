# import streamlit as st
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import google.generativeai as genai

# # --- Configuration and Setup ---
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# FAISS_INDEX_PATH = "faiss_index"

# # --- Caching to improve performance ---
# @st.cache_resource
# def load_vector_store():
#     """Loads the FAISS index from disk."""
#     print("Loading vector store...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     db = FAISS.load_local(
#         FAISS_INDEX_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )
#     print("Vector store loaded successfully.")
#     return db

# @st.cache_resource
# def get_conversational_chain():
#     """Creates and returns the QA chain."""
#     prompt_template = """
#     You are a helpful legal assistant. Answer the question as detailed as possible from the provided context.
#     Make sure to provide all the details. If the answer is not in the provided context, just say,
#     "The answer is not available in the provided documents." Do not provide a wrong answer.\n\n
#     Context:\n {context}\n
#     Question:\n {question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def main():
#     """Main function to run the Streamlit app."""
#     st.set_page_config(page_title="Supreme Court RAG", page_icon="⚖️")
#     st.header("⚖️ Supreme Court Judgment Chatbot")

#     # --- Initial Check ---
#     if not os.path.exists(FAISS_INDEX_PATH):
#         st.error("Vector store index not found!")
#         st.info("Please run `1_ingest_data.py` first to create the knowledge base.")
#         st.stop()

#     # --- Load Resources ---
#     db = load_vector_store()
#     chain = get_conversational_chain()

#     # --- User Interface ---
#     user_question = st.text_input("Ask a question about the Supreme Court judgments:")

#     if st.button("Get Answer") and user_question:
#         with st.spinner("Searching the archives..."):
#             # Perform search and get relevant documents
#             docs = db.similarity_search(user_question, k=5) # Get top 5 relevant chunks

#             if not docs:
#                 st.warning("Could not find any relevant documents for your question.")
#                 return

#             # Get the response from the LLM
#             response = chain(
#                 {"input_documents": docs, "question": user_question},
#                 return_only_outputs=True
#             )

#             # --- Display Results ---
#             st.subheader("Answer:")
#             st.write(response["output_text"])

#             with st.expander("Show Sources"):
#                 st.write("The answer was generated based on the following documents:")
#                 # Using a set to avoid duplicate sources if multiple chunks from the same file are used
#                 unique_sources = set()
#                 for doc in docs:
#                     source_info = (
#                         doc.metadata.get('source', 'Unknown'),
#                         doc.metadata.get('year', 'N/A')
#                     )
#                     unique_sources.add(source_info)
                
#                 for source, year in sorted(list(unique_sources)):
#                     st.markdown(f"- **File:** `{source}` | **Year:** `{year}`")

# if __name__ == "__main__":
#     main()

# new

# File: 2_chat_with_pdfs.py

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration and Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
FAISS_INDEX_PATH = "faiss_index"

# --- Caching to improve performance ---
@st.cache_resource
def load_vector_store():
    """Loads the FAISS index from disk. This is cached for performance."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"Index not found at '{FAISS_INDEX_PATH}'. Please run '1_build_database.py' first.")
        st.stop()
    
    print("Loading vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
    return db

@st.cache_resource
def get_conversational_chain():
    """Creates and returns the QA chain. This is cached for performance."""
    prompt_template = """
    You are a specialized legal assistant. Your task is to answer questions based *only* on the provided context from Supreme Court documents.
    Answer in detail, providing all relevant information found in the context.
    If the answer is not present in the provided context, you MUST state: "The answer is not available in the provided documents."
    Do not, under any circumstances, invent an answer or use external knowledge.

    Context:\n {context}\n
    Question:\n {question}\n

    Detailed Answer:
    """
    # This is the correct model name for chat, without the "models/" prefix.
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Supreme Court RAG", page_icon="⚖️")
    st.header("⚖️ Supreme Court Judgment Chatbot")

    # --- Load Resources ---
    db = load_vector_store()
    chain = get_conversational_chain()

    # --- User Interface ---
    user_question = st.text_input("Ask a question about the Supreme Court judgments:")

    if st.button("Get Answer") and user_question:
        with st.spinner("Searching the legal archives..."):
            # Perform search and get relevant documents
            docs = db.similarity_search(user_question, k=5) 

            if not docs:
                st.warning("Could not find any relevant documents for your question.")
                return

            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )

            # --- Display Results ---
            st.subheader("Answer:")
            st.write(response["output_text"])

            with st.expander("Show Sources Used"):
                st.write("The answer was generated based on information from the following documents:")
                unique_sources = set((doc.metadata.get('source', 'Unknown'), doc.metadata.get('year', 'N/A')) for doc in docs)
                for source, year in sorted(list(unique_sources)):
                    st.markdown(f"- **File:** `{source}` | **Year:** `{year}`")

if __name__ == "__main__":
    main()
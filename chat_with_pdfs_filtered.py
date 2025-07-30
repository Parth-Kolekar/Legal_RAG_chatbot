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
        st.error(f"Index not found at '{FAISS_INDEX_PATH}'. Please run the data ingestion script first.")
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
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# --- NEW FEATURE: Metadata-based Filtering ---
def get_relevant_documents(db, query, start_year, end_year):
    """
    Performs a similarity search with a pre-filter based on the year metadata.
    """
    # FAISS filter format: a function that takes a dictionary (metadata) and returns a boolean.
    # We will only include documents where the 'year' metadata falls within the selected range.
    def year_filter(metadata: dict) -> bool:
        try:
            # Safely get the year and convert to integer for comparison
            doc_year = int(metadata.get("year", 0))
            return start_year <= doc_year <= end_year
        except (ValueError, TypeError):
            # If 'year' is not a valid number or is missing, exclude it from the search
            return False

    # The similarity_search method in some vectorstores supports a 'filter' argument.
    # For FAISS, we use the more advanced `similarity_search_with_relevance_scores` 
    # and then apply the filter manually, as direct filtering isn't a standard FAISS feature
    # but LangChain provides workarounds. However, the most robust way is to use a 
    # vector store that natively supports metadata filtering like Chroma or Pinecone.
    
    # Let's implement the most compatible approach for FAISS: search then filter.
    # While not as efficient as a pre-filter, it's the standard LangChain way for FAISS.
    
    # A more advanced vector store would do this:
    # return db.similarity_search(query, k=5, filter=year_filter)

    # For FAISS, we get more results and then filter them down.
    # This is less efficient than a true pre-filter but works reliably with FAISS.
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 20} # Retrieve more documents to increase chance of finding matches in the date range
    )
    
    all_docs = retriever.get_relevant_documents(query)
    
    # Now, filter these documents by the year range
    filtered_docs = [doc for doc in all_docs if year_filter(doc.metadata)]
    
    # Return the top 5 from the filtered list
    return filtered_docs[:5]


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Supreme Court RAG", page_icon="⚖️")
    st.header("⚖️ Supreme Court Judgment Chatbot")

    # --- Load Resources ---
    db = load_vector_store()
    chain = get_conversational_chain()

    # --- NEW: Sidebar for Filtering ---
    with st.sidebar:
        st.title("Search Filters")
        st.write("Select a range of years to search within:")
        
        # We assume your data is from 1950 to 2025 as you mentioned
        start_year, end_year = st.slider(
            "Year Range",
            min_value=1950,
            max_value=2025,
            value=(1950, 2025) # Default to the full range
        )

    # --- User Interface ---
    user_question = st.text_input("Ask a question about the Supreme Court judgments:")

    if st.button("Get Answer") and user_question:
        with st.spinner(f"Searching archives from {start_year} to {end_year}..."):
            
            # --- MODIFIED: Use the new filtering function ---
            docs = get_relevant_documents(db, user_question, start_year, end_year)

            if not docs:
                st.warning(f"Could not find any relevant documents for your question within the years {start_year}-{end_year}.")
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
# import os
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv
# import google.generativeai as genai
# import time

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # --- Configuration ---
# SOURCE_DIRECTORY = "pdfs"
# FAISS_INDEX_PATH = "faiss_index"
# CHUNK_SIZE = 10000
# CHUNK_OVERLAP = 1000

# def load_documents_from_directory(directory_path):
#     """
#     Loads all PDF files from a nested directory structure, extracting text
#     and attaching metadata (source file and year).
#     """
#     documents = []
#     print(f"Scanning directory: {directory_path}")

#     for root, _, files in os.walk(directory_path):
#         # Extract the year from the folder path
#         # os.path.basename will get the last part of the path (e.g., '1950')
#         year = os.path.basename(root)
#         if not year.isdigit(): # Skip the root folder or non-year folders
#             year = None

#         for file in files:
#             if file.lower().endswith(".pdf"):
#                 file_path = os.path.join(root, file)
#                 print(f"  > Processing: {file_path}")
#                 try:
#                     pdf_reader = PdfReader(file_path)
#                     text = ""
#                     for page in pdf_reader.pages:
#                         text += page.extract_text() or ""
                    
#                     if text:
#                         # Create a Document object for each file
#                         documents.append(Document(
#                             page_content=text,
#                             metadata={
#                                 'source': file,
#                                 'year': year if year else 'N/A',
#                                 'full_path': file_path
#                             }
#                         ))
#                 except Exception as e:
#                     print(f"    [!] Error reading {file_path}: {e}")
    
#     print(f"\nSuccessfully loaded {len(documents)} documents.")
#     return documents

# def split_documents_into_chunks(documents):
#     """Splits the loaded documents into smaller chunks for processing."""
#     print("Splitting documents into chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Total chunks created: {len(chunks)}")
#     return chunks

# def create_and_save_vector_store(chunks):
#     """Creates embeddings and saves them to a local FAISS index."""
#     if not chunks:
#         print("No chunks to process. Exiting.")
#         return

#     print("Initializing embeddings model (models/embedding-001)...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     print("Creating FAISS vector store from chunks. This can take a long time...")
#     start_time = time.time()
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#     end_time = time.time()
    
#     print(f"Vector store created in {end_time - start_time:.2f} seconds.")
    
#     print(f"Saving vector store to: {FAISS_INDEX_PATH}")
#     vector_store.save_local(FAISS_INDEX_PATH)
#     print("Vector store saved successfully!")


# def main():
#     """Main function to run the data ingestion pipeline."""
#     print("--- Starting Data Ingestion and Vector DB Creation ---")
    
#     if not os.path.isdir(SOURCE_DIRECTORY):
#         print(f"Error: Source directory '{SOURCE_DIRECTORY}' not found.")
#         return
        
#     # 1. Load documents with metadata
#     documents = load_documents_from_directory(SOURCE_DIRECTORY)
    
#     if documents:
#         # 2. Split documents into chunks
#         chunks = split_documents_into_chunks(documents)
        
#         # 3. Create and save the vector store
#         create_and_save_vector_store(chunks)
#     else:
#         print("No documents were loaded. Halting process.")
        
#     print("--- Data Ingestion Complete ---")


# if __name__ == "__main__":
#     main()


# File: 1_build_database.py

import os
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# IMPORTANT: Set your source folder here
PDF_SOURCE_FOLDER = "pdfs/supreme_court_judgments/2019" 
FAISS_INDEX_PATH = "faiss_index"

def load_documents_from_folders(directory_path):
    """
    Scans all subfolders for PDF files, extracts text, and creates Document objects
    with metadata (source filename and year).
    """
    documents = []
    print(f"[*] Starting to scan PDF files in '{directory_path}'...")

    for root, _, files in os.walk(directory_path):
        year = os.path.basename(root)
        if not year.isdigit():
            year = 'Unknown' # Handle the root folder or other non-year folders

        for file_name in files:
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(root, file_name)
                try:
                    print(f"  - Reading: {file_name} (Year: {year})")
                    pdf_reader = PdfReader(file_path)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                    
                    if text:
                        documents.append(Document(
                            page_content=text,
                            metadata={'source': file_name, 'year': year}
                        ))
                except Exception as e:
                    print(f"    [!] Error processing {file_name}: {e}")
    
    print(f"\n[SUCCESS] Loaded a total of {len(documents)} documents.")
    return documents

def build_and_save_vector_store(documents):
    """Splits documents, creates embeddings, and saves them to a FAISS index."""
    if not documents:
        print("[ERROR] No documents were loaded. Cannot build vector store.")
        return

    print("[*] Splitting documents into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(documents)
    print(f"[*] Created {len(chunks)} text chunks.")

    print("[*] Initializing Google's embedding model...")
    # This model name IS correct and requires the "models/" prefix
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print("[*] Creating FAISS vector store from chunks. This will take some time...")
    start_time = time.time()
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    end_time = time.time()
    
    print(f"[*] Vector store created in {end_time - start_time:.2f} seconds.")
    
    print(f"[*] Saving vector store to local disk at '{FAISS_INDEX_PATH}'...")
    vector_store.save_local(FAISS_INDEX_PATH)
    print("\n[SUCCESS] Vector database has been built and saved.")

def main():
    """Main function to run the data ingestion and database creation pipeline."""
    if not os.path.isdir(PDF_SOURCE_FOLDER):
        print(f"[FATAL ERROR] The source folder '{PDF_SOURCE_FOLDER}' was not found.")
        print("Please make sure it exists and contains your PDF files.")
        return
        
    documents = load_documents_from_folders(PDF_SOURCE_FOLDER)
    build_and_save_vector_store(documents)

if __name__ == "__main__":
    main()
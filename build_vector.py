# # import os
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.docstore.document import Document
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from dotenv import load_dotenv
# # import google.generativeai as genai
# # import time

# # # Load environment variables
# # load_dotenv()
# # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # # --- Configuration ---
# # SOURCE_DIRECTORY = "pdfs"
# # FAISS_INDEX_PATH = "faiss_index"
# # CHUNK_SIZE = 10000
# # CHUNK_OVERLAP = 1000

# # def load_documents_from_directory(directory_path):
# #     """
# #     Loads all PDF files from a nested directory structure, extracting text
# #     and attaching metadata (source file and year).
# #     """
# #     documents = []
# #     print(f"Scanning directory: {directory_path}")

# #     for root, _, files in os.walk(directory_path):
# #         # Extract the year from the folder path
# #         # os.path.basename will get the last part of the path (e.g., '1950')
# #         year = os.path.basename(root)
# #         if not year.isdigit(): # Skip the root folder or non-year folders
# #             year = None

# #         for file in files:
# #             if file.lower().endswith(".pdf"):
# #                 file_path = os.path.join(root, file)
# #                 print(f"  > Processing: {file_path}")
# #                 try:
# #                     pdf_reader = PdfReader(file_path)
# #                     text = ""
# #                     for page in pdf_reader.pages:
# #                         text += page.extract_text() or ""
                    
# #                     if text:
# #                         # Create a Document object for each file
# #                         documents.append(Document(
# #                             page_content=text,
# #                             metadata={
# #                                 'source': file,
# #                                 'year': year if year else 'N/A',
# #                                 'full_path': file_path
# #                             }
# #                         ))
# #                 except Exception as e:
# #                     print(f"    [!] Error reading {file_path}: {e}")
    
# #     print(f"\nSuccessfully loaded {len(documents)} documents.")
# #     return documents

# # def split_documents_into_chunks(documents):
# #     """Splits the loaded documents into smaller chunks for processing."""
# #     print("Splitting documents into chunks...")
# #     text_splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=CHUNK_SIZE,
# #         chunk_overlap=CHUNK_OVERLAP,
# #     )
# #     chunks = text_splitter.split_documents(documents)
# #     print(f"Total chunks created: {len(chunks)}")
# #     return chunks

# # def create_and_save_vector_store(chunks):
# #     """Creates embeddings and saves them to a local FAISS index."""
# #     if not chunks:
# #         print("No chunks to process. Exiting.")
# #         return

# #     print("Initializing embeddings model (models/embedding-001)...")
# #     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# #     print("Creating FAISS vector store from chunks. This can take a long time...")
# #     start_time = time.time()
# #     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
# #     end_time = time.time()
    
# #     print(f"Vector store created in {end_time - start_time:.2f} seconds.")
    
# #     print(f"Saving vector store to: {FAISS_INDEX_PATH}")
# #     vector_store.save_local(FAISS_INDEX_PATH)
# #     print("Vector store saved successfully!")


# # def main():
# #     """Main function to run the data ingestion pipeline."""
# #     print("--- Starting Data Ingestion and Vector DB Creation ---")
    
# #     if not os.path.isdir(SOURCE_DIRECTORY):
# #         print(f"Error: Source directory '{SOURCE_DIRECTORY}' not found.")
# #         return
        
# #     # 1. Load documents with metadata
# #     documents = load_documents_from_directory(SOURCE_DIRECTORY)
    
# #     if documents:
# #         # 2. Split documents into chunks
# #         chunks = split_documents_into_chunks(documents)
        
# #         # 3. Create and save the vector store
# #         create_and_save_vector_store(chunks)
# #     else:
# #         print("No documents were loaded. Halting process.")
        
# #     print("--- Data Ingestion Complete ---")


# # if __name__ == "__main__":
# #     main()


# # File: 1_build_database.py

# import os
# import time
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv
# import google.generativeai as genai

# # --- Configuration ---
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # IMPORTANT: Set your source folder here
# PDF_SOURCE_FOLDER = "pdfs/supreme_court_judgments/2019" 
# FAISS_INDEX_PATH = "faiss_index"

# def load_documents_from_folders(directory_path):
#     """
#     Scans all subfolders for PDF files, extracts text, and creates Document objects
#     with metadata (source filename and year).
#     """
#     documents = []
#     print(f"[*] Starting to scan PDF files in '{directory_path}'...")

#     for root, _, files in os.walk(directory_path):
#         year = os.path.basename(root)
#         if not year.isdigit():
#             year = 'Unknown' # Handle the root folder or other non-year folders

#         for file_name in files:
#             if file_name.lower().endswith(".pdf"):
#                 file_path = os.path.join(root, file_name)
#                 try:
#                     print(f"  - Reading: {file_name} (Year: {year})")
#                     pdf_reader = PdfReader(file_path)
#                     text = ""
#                     for page in pdf_reader.pages:
#                         text += page.extract_text() or ""
                    
#                     if text:
#                         documents.append(Document(
#                             page_content=text,
#                             metadata={'source': file_name, 'year': year}
#                         ))
#                 except Exception as e:
#                     print(f"    [!] Error processing {file_name}: {e}")
    
#     print(f"\n[SUCCESS] Loaded a total of {len(documents)} documents.")
#     return documents

# def build_and_save_vector_store(documents):
#     """Splits documents, creates embeddings, and saves them to a FAISS index."""
#     if not documents:
#         print("[ERROR] No documents were loaded. Cannot build vector store.")
#         return

#     print("[*] Splitting documents into manageable chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_documents(documents)
#     print(f"[*] Created {len(chunks)} text chunks.")

#     print("[*] Initializing Google's embedding model...")
#     # This model name IS correct and requires the "models/" prefix
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     print("[*] Creating FAISS vector store from chunks. This will take some time...")
#     start_time = time.time()
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#     end_time = time.time()
    
#     print(f"[*] Vector store created in {end_time - start_time:.2f} seconds.")
    
#     print(f"[*] Saving vector store to local disk at '{FAISS_INDEX_PATH}'...")
#     vector_store.save_local(FAISS_INDEX_PATH)
#     print("\n[SUCCESS] Vector database has been built and saved.")

# def main():
#     """Main function to run the data ingestion and database creation pipeline."""
#     if not os.path.isdir(PDF_SOURCE_FOLDER):
#         print(f"[FATAL ERROR] The source folder '{PDF_SOURCE_FOLDER}' was not found.")
#         print("Please make sure it exists and contains your PDF files.")
#         return
        
#     documents = load_documents_from_folders(PDF_SOURCE_FOLDER)
#     build_and_save_vector_store(documents)

# if __name__ == "__main__":
#     main()


################################################################################################
# import os
# import time
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.docstore.document import Document
# from langchain.embeddings import OpenAIEmbeddings

# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv

# # --- Configuration ---
# load_dotenv()

# # IMPORTANT: Set your source folder here
# PDF_SOURCE_FOLDER = "pdfs/test_data"
# FAISS_INDEX_PATH = "faiss_index"

# def load_documents_from_folders(directory_path):
#     """
#     Scans all subfolders for PDF files, extracts text, and creates Document objects
#     with metadata (source filename and year).
#     """
#     documents = []
#     print(f"[*] Starting to scan PDF files in '{directory_path}'...")

#     for root, _, files in os.walk(directory_path):
#         year = os.path.basename(root)
#         if not year.isdigit():
#             year = 'Unknown'

#         for file_name in files:
#             if file_name.lower().endswith(".pdf"):
#                 file_path = os.path.join(root, file_name)
#                 try:
#                     print(f"  - Reading: {file_name} (Year: {year})")
#                     pdf_reader = PdfReader(file_path)
#                     text = ""
#                     for page in pdf_reader.pages:
#                         text += page.extract_text() or ""
                    
#                     if text:
#                         documents.append(Document(
#                             page_content=text,
#                             metadata={'source': file_name, 'year': year}
#                         ))
#                 except Exception as e:
#                     print(f"    [!] Error processing {file_name}: {e}")
    
#     print(f"\n[SUCCESS] Loaded a total of {len(documents)} documents.")
#     return documents

# def build_and_save_vector_store(documents):
#     """Splits documents, creates embeddings, and saves them to a FAISS index."""
#     if not documents:
#         print("[ERROR] No documents were loaded. Cannot build vector store.")
#         return

#     print("[*] Splitting documents into manageable chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_documents(documents)
#     print(f"[*] Created {len(chunks)} text chunks.")


#     print("[*] Initializing OpenRouter's embedding model...")
#     # embeddings = OpenAIEmbeddings(
#     #     model="text-embedding-3-large",
#     #     openai_api_base="https://openrouter.ai/api/v1",
#     #     openai_api_key=os.getenv("OPENROUTER_API_KEY")
#     # )
#     embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     openai_api_key=os.getenv("OPENAI_API_KEY")  # MUST be OpenAI key
#     )


#     print("[*] Creating FAISS vector store from chunks. This will take some time...")
#     start_time = time.time()
#     vector_store = FAISS.from_documents(chunks, embedding=embeddings)
#     end_time = time.time()
    
#     print(f"[*] Vector store created in {end_time - start_time:.2f} seconds.")
    
#     print(f"[*] Saving vector store to local disk at '{FAISS_INDEX_PATH}'...")
#     vector_store.save_local(FAISS_INDEX_PATH)
#     print("\n[SUCCESS] Vector database has been built and saved.")

# def main():
#     """Main function to run the data ingestion and database creation pipeline."""
#     if not os.path.isdir(PDF_SOURCE_FOLDER):
#         print(f"[FATAL ERROR] The source folder '{PDF_SOURCE_FOLDER}' was not found.")
#         print("Please make sure it exists and contains your PDF files.")
#         return
        
#     documents = load_documents_from_folders(PDF_SOURCE_FOLDER)
#     build_and_save_vector_store(documents)

# if __name__ == "__main__":
#     main()


##########################################################################33import os
"""
build_vector.py
===============
Reads Supreme Court judgment PDFs (Indian Kanoon format),
extracts text + metadata, and builds a FAISS index.

YOUR FOLDER STRUCTURE should look like:
    pdfs/
        2019/
            Case_Name_1.pdf
            Case_Name_2.pdf
        2020/
            ...
        2023/
            Abhishek_vs_State_of_MP.pdf

USAGE:
    python build_vector.py

Run once on your laptop. Commit faiss_index/ to GitHub.
Streamlit Cloud loads it directly — no rebuild needed on server.
"""

import os
import re
import time
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_FOLDER       = "pdfs/supreme_court_judgments/2025"          # your root folder containing year subfolders
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
BATCH_SIZE    = 50

# Set to a number like 50 to test first, then set to None for all
MAX_PDFS = 50
# ─────────────────────────────────────────────────────────────────────────────


def extract_metadata_from_text(text: str, filename: str, folder_year: str) -> dict:
    """
    Indian Kanoon PDFs always have this header format at the top:
        Case Name vs Other Party on DD Month, YYYY
        Author: Judge Name
        Bench: Judge1, Judge2
    We extract that cleanly.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    title  = lines[0] if lines else filename
    author = ""
    bench  = ""
    year   = folder_year  # default to folder name

    for line in lines[:10]:
        if line.startswith("Author:"):
            author = line.replace("Author:", "").strip()
        elif line.startswith("Bench:"):
            bench = line.replace("Bench:", "").strip()

    # Extract year from title line e.g. "... on 31 August, 2023"
    year_match = re.search(r'\b(20\d{2}|199\d)\b', title)
    if year_match:
        year = year_match.group(1)

    return {
        "title":  title[:120],
        "author": author,
        "bench":  bench,
        "year":   year,
        "source": filename,
    }


def load_pdfs(root_folder: str) -> list[Document]:
    """
    Walk the folder structure, extract text from each PDF,
    attach metadata, return list of Documents.
    """
    if not os.path.isdir(root_folder):
        print(f"[ERROR] Folder '{root_folder}' not found.")
        print(f"  Make sure your PDFs are in: {root_folder}/YEAR/filename.pdf")
        return []

    documents = []
    errors    = 0
    count     = 0

    # Collect all PDF paths first
    all_pdfs = []
    for root, _, files in os.walk(root_folder):
        year = os.path.basename(root)
        year = year if re.match(r'^(19|20)\d{2}$', year) else "Unknown"
        for f in files:
            if f.lower().endswith(".pdf"):
                all_pdfs.append((os.path.join(root, f), f, year))

    total = len(all_pdfs) if MAX_PDFS is None else min(len(all_pdfs), MAX_PDFS)
    print(f"[*] Found {len(all_pdfs)} PDFs. Processing {total}.\n")

    for filepath, filename, folder_year in all_pdfs[:total]:
        count += 1
        print(f"  [{count}/{total}] {filename[:65]}", end=" ... ")

        try:
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            text = text.strip()

            if len(text) < 200:
                print("skip (too short)")
                errors += 1
                continue

            meta = extract_metadata_from_text(text, filename, folder_year)

            # Remove the duplicate title line that Indian Kanoon PDFs have at top
            # (first line is repeated twice)
            lines = text.split('\n')
            if len(lines) > 1 and lines[0].strip() == lines[1].strip():
                text = '\n'.join(lines[1:])

            documents.append(Document(page_content=text, metadata=meta))
            print(f"✓  ({len(text):,} chars, {meta['year']})")

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    print(f"\n[*] Loaded {len(documents)} documents. ({errors} skipped/errors)")
    return documents


def build_and_save_index(documents: list[Document]):
    """Split docs into chunks, embed locally, save FAISS index."""

    print(f"\n[*] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[*] {len(chunks)} chunks from {len(documents)} documents.")

    print(f"\n[*] Loading embedding model '{EMBEDDING_MODEL}'...")
    print("  First run downloads ~90MB. Cached after that.")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"\n[*] Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    print("  ~1-2 min per 100 chunks on CPU.\n")

    start         = time.time()
    vector_store  = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch     = chunks[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.merge_from(FAISS.from_documents(batch, embeddings))
        print("done")

    elapsed = time.time() - start
    print(f"\n[*] Embedding complete in {elapsed:.0f}s ({elapsed/60:.1f} min).")

    print(f"[*] Saving to '{FAISS_INDEX_PATH}'...")
    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"""
✅ Done! Index saved to '{FAISS_INDEX_PATH}/'

  Documents : {len(documents)}
  Chunks    : {len(chunks)}
  Time      : {elapsed/60:.1f} min

Next steps:
  Local  →  streamlit run app.py
  Deploy →  commit '{FAISS_INDEX_PATH}/' to GitHub, deploy on Streamlit Cloud
""")


def main():
    documents = load_pdfs(PDF_FOLDER)
    if not documents:
        print("\nNo documents loaded. Check your folder structure.")
        return
    build_and_save_index(documents)


if __name__ == "__main__":
    main()
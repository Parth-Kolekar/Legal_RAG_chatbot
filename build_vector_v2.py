"""
build_vector_hf.py  —  Run ONCE on your laptop
================================================
Downloads the HuggingFace dataset (21k chunks from ~100 SC judgments),
builds a FAISS index, saves it to disk.

Usage:
    pip install -r requirements.txt
    python build_vector_hf.py
"""

import time
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "faiss_index_v2"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE       = 100   # bigger batches ok since chunks are pre-made

print("[1/3] Downloading dataset from HuggingFace...")
print("      (~50MB, takes 1-2 min on first run, cached after)\n")

ds = load_dataset("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", split="train")
print(f"      Loaded {len(ds)} chunks.\n")

# Convert to LangChain Documents
# Dataset only has 'text' column — we add minimal metadata
docs = [
    Document(
        page_content=row["text"],
        metadata={"source": f"chunk_{i}", "row": i}
    )
    for i, row in enumerate(ds)
    if len(row["text"].strip()) > 50   # skip near-empty rows
]
print(f"[2/3] {len(docs)} valid chunks ready for embedding.\n")

print(f"[3/3] Loading embedding model '{EMBEDDING_MODEL}'...")
print("      (Downloads ~90MB on first run, then cached)\n")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print(f"Embedding {len(docs)} chunks in batches of {BATCH_SIZE}...")
print("~2-4 min on CPU. Please wait.\n")

start        = time.time()
vector_store = None
total        = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i: i + BATCH_SIZE]
    n     = i // BATCH_SIZE + 1
    print(f"  Batch {n}/{total}...", end=" ", flush=True)
    if vector_store is None:
        vector_store = FAISS.from_documents(batch, embeddings)
    else:
        vector_store.merge_from(FAISS.from_documents(batch, embeddings))
    print("done")

elapsed = time.time() - start
vector_store.save_local(FAISS_INDEX_PATH)

print(f"""
✅ Done in {elapsed/60:.1f} min.
   Index saved → '{FAISS_INDEX_PATH}/'

Next:  streamlit run app.py
Deploy: commit '{FAISS_INDEX_PATH}/' to GitHub → Streamlit Cloud
""")
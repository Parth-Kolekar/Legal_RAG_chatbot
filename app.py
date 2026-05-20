import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
GEMINI_MODEL     = "gemini-3.5-flash"
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_chain():
    """
    Load the vector store and set up the conversational chain.
    Cached so it only runs once per session.
    """
    # 1. Load embeddings (same model used in build_vector.py)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # embeddings = HuggingFaceEmbeddings(
    # model_name=EMBEDDING_MODEL,
    # model_kwargs={"device": "cuda"},
    # encode_kwargs={"normalize_embeddings": True},
    # )  
    # for gpu

    # 2. Load FAISS index from disk
    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    # 3. Gemini as LLM — reads GOOGLE_API_KEY from .env or Streamlit secrets
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True,
    )

    # 4. Memory — keeps last 5 exchanges so follow-up questions work
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # 5. ConversationalRetrievalChain — handles follow-up questions naturally
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def format_sources(source_docs: list) -> str:
    """Format source documents into a readable citation list."""
    seen = set()
    lines = []
    for doc in source_docs:
        title = doc.metadata.get("title", doc.metadata.get("source", "Unknown"))
        year  = doc.metadata.get("year", "N/A")
        key   = (title, year)
        if key not in seen:
            seen.add(key)
            lines.append(f"• **{title[:80]}** ({year})")
    return "\n".join(lines) if lines else "Sources not available."


# ── PAGE SETUP ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SC Judgment Chatbot",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ Supreme Court Judgment Chatbot")
st.caption("Ask questions about Indian Supreme Court judgments (2025). "
           "Powered by Gemini 3.5 Flash + FAISS.")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info(
        "This chatbot can answer questions based on Supreme Court judgments.\n\n"
        "**Tip:** Ask follow-up questions — it remembers the conversation context.\n\n"
        "**Examples:**\n"
        "- What is the right to life under Article 21?\n"
        "- What factors does the court consider for bail?\n"
        "- Explain the principle of natural justice."
    )

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Data source: Indian Kanoon API\nLLM: Gemini 3.5 Flash (free tier)")

# ── LOAD CHAIN ────────────────────────────────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("⚠️ Vector index not found!")
    st.info(
        "You need to build the knowledge base first:\n\n"
        "```bash\n"
        "python fetch_judgments.py   # download judgments\n"
        "python build_vector.py      # build the index\n"
        "```"
    )
    st.stop()

chain = load_chain()

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help you understand Supreme Court judgments. "
                       "What would you like to know?",
        }
    ]

# Display all prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Sources used"):
                st.markdown(msg["sources"])

# ── HANDLE NEW INPUT ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about SC judgments..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching judgments..."):
            try:
                result   = chain({"question": prompt})
                answer   = result.get("answer", "Sorry, I couldn't find an answer.")
                src_docs = result.get("source_documents", [])
                sources  = format_sources(src_docs)

                st.markdown(answer)
                if src_docs:
                    with st.expander("📄 Sources used"):
                        st.markdown(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                err_msg = f"Error: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                })
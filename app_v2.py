
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

load_dotenv()

FAISS_INDEX_PATH = "faiss_index_v2"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


# ── Load everything once and cache ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},#cuda for gpu
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings,
        allow_dangerous_deserialization=True,
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True,
    )
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SC Judgment Chatbot", page_icon="⚖️", layout="centered")
st.title("⚖️ Supreme Court Judgment Chatbot")
st.caption("Ask anything about Indian Supreme Court judgments. Follows up questions are remembered.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try asking")
    st.markdown("""
- What did the court say about right to privacy?
- Explain the Aadhaar judgment
- What is Section 498A IPC?
- When can an FIR be quashed?
- What are the bail conditions set by SC?
- Tell me about the triple talaq verdict
- What is natural justice principle?
""")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.memory_reset = True
        st.rerun()
    st.caption("Memory: last 6 exchanges\nLLM: Gemini 3.5 Flash (free)\nData: ~100 SC judgments")

# ── Guard: index must exist ───────────────────────────────────────────────────
if not os.path.exists(FAISS_INDEX_PATH):
    st.error("⚠️ Vector index not found!")
    st.code("python build_vector_hf.py", language="bash")
    st.info("Run the above command once to build the knowledge base.")
    st.stop()

chain = load_chain()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I can answer questions about Supreme Court judgments. "
                   "You can ask follow-up questions — I remember our conversation. What would you like to know?"
    }]

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source chunks used"):
                st.markdown(msg["sources"])

# ── Handle new input ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about any SC judgment..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching judgments..."):
            try:
                result   = chain({"question": prompt})
                answer   = result.get("answer", "Sorry, I couldn't find a relevant answer.")
                src_docs = result.get("source_documents", [])

                # Format sources — show first ~200 chars of each unique chunk
                source_lines = []
                seen = set()
                for doc in src_docs:
                    preview = doc.page_content[:200].replace("\n", " ").strip()
                    if preview not in seen:
                        seen.add(preview)
                        source_lines.append(f"• ...{preview}...")
                sources_text = "\n\n".join(source_lines) if source_lines else ""

                st.markdown(answer)
                if sources_text:
                    with st.expander("📄 Source chunks used"):
                        st.markdown(sources_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text,
                })

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
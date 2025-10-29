# streamlit_agentic_ai.py

import os
import streamlit as st
from dotenv import load_dotenv
from typing import List, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from typing import Literal

# ------------------- ENV SETUP -------------------
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["HF_API_KEY"] = os.getenv("HF_API_KEY", "")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

# ------------------- IMPORTS -------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langsmith import Client

# ------------------- PAGE + STYLE (Professional Minimal) -------------------
st.set_page_config(page_title="Agentic AI Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
      /* Container */
      .block-container { padding-left:2rem; padding-right:2rem; padding-top:1.2rem; }
      /* Sidebar */
      [data-testid="stSidebar"] { background-color: #ffffff; border-right:1px solid #e6e6e6; }
      /* Headings */
      h1 { color:#0f172a; font-weight:700; }
      /* Chat bubbles: user (subtle blue) and assistant (light gray) */
      .user-bubble { background:#e8f0fe; border-radius:12px; padding:10px 14px; margin:6px 0; color:#0b3d91; }
      .assistant-bubble { background:#f7f7fb; border-radius:12px; padding:10px 14px; margin:6px 0; color:#0f172a; }
      /* Small meta */
      .meta { color: #6b7280; font-size:12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- SESSION STATE -------------------
if "chat_history" not in st.session_state:
    # chat_history is list of dicts: {"role":"user"/"assistant", "content": str}
    st.session_state["chat_history"] = []

if "route" not in st.session_state:
    st.session_state["route"] = None

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

# ------------------- SIDEBAR: HISTORY + Upload + Controls -------------------
with st.sidebar:
    st.header("💬 Conversation")
    if st.session_state["chat_history"]:
        # Show only user turns as titles (3-4 words)
        user_turns = [c for c in st.session_state["chat_history"] if c["role"] == "user"]
        for i, turn in enumerate(user_turns, start=1):
            short = " ".join(turn["content"].split()[:4])
            st.write(f"**{i}.** {short}...")
    else:
        st.info("No conversation yet — start by typing below.")

    st.markdown("---")
    st.subheader("Knowledge base")
    st.write("Upload PDFs (optional). If none uploaded, router will avoid vectorstore.")
    uploaded_files = st.file_uploader(
        label="Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True,
        help="Optional: upload PDFs to build a local knowledge base used by vectorstore/RAG."
    )

    st.markdown("---")
    st.subheader("Modes & Controls")
    mode = st.radio("Preferred mode:", options=["Work Only", "Normal Chat only"], index=0,
                    help="Agentic auto-routes between vectorstore/web/chat. Normal Chat uses only the LLM.")
    if st.button("Clear chat"):
        st.session_state["chat_history"] = []
        st.session_state["route"] = None
        st.experimental_rerun()

    st.markdown("---")
    st.caption("Built with Power of Agentic AI.")

# ------------------- MAIN UI HEADER -------------------
st.title("🤖 AI Assistant — Affordable Academic Writing ")
st.write("Ask questions naturally & Make Writing Natural.")
st.divider()

# ------------------- OPTIONAL: Build / Update Vectorstore from uploaded PDFs -------------------
if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
            try:
                os.remove(temp_path)
            except Exception:
                pass

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        st.session_state["vectorstore"] = vectorstore
        st.session_state["retriever"] = retriever
    st.success("PDF knowledge base ready ✔")
else:
    # No files uploaded; ensure retriever is None
    st.session_state["vectorstore"] = None
    st.session_state["retriever"] = None

# ------------------- TOOLS: Tavily -------------------
web_search_tool = TavilySearch(k=1)

# ------------------- LLM & Router Setup -------------------
class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""
    datasource: Literal["vectorstore", "web_search", "chat"] = Field(
        ..., description="Which datasource to use"
    )

# Primary LLM (Gemini via ChatGoogleGenerativeAI)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
structured_llm_router = llm.with_structured_output(RouteQuery)

system_router = (
    "You are an expert router. Given a user question decide whether the best source is:\n"
    "- 'vectorstore' (use when question is clearly about uploaded PDFs),\n"
    "- 'web_search' (use for general web/external info),\n"
    "- 'chat' (use when user wants generic chat, creative or open-ended conversation).\n"
    "If user asks for document-specific answers, prefer vectorstore. If PDFs are not available, fallback to web_search or chat."
)

route_prompt = ChatPromptTemplate([
    ("system", system_router),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

# ------------------- RAG CHAIN (prompt pulled from LangSmith if available) -------------------
client = Client()
try:
    prompt = client.pull_prompt("rlm/rag-prompt")
except Exception:
    # Fallback simple prompt if pull fails (keeps code robust)
    from langchain_core.prompts import PromptTemplate
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use the provided context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely."
    )
# Compose rag_chain: prompt -> llm -> string parser
try:
    rag_chain = prompt | llm | StrOutputParser()
except Exception:
    # If chaining operator unavailable/fails, fallback to llm usage directly in generate node
    rag_chain = None

# ------------------- Graph State Type -------------------
class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    documents: List[Document]
    chat_history: List[dict]

# ------------------- Graph Node Implementations -------------------
def retrieve(state: dict):
    """
    Retrieve documents from vectorstore (FAISS). Returns docs and flows to generate.
    This function is robust to different retriever APIs.
    """
    st.session_state["route"] = "vectorstore"
    question = state["question"]
    retriever = st.session_state.get("retriever")

    if not retriever:
        # Fallback - if retriever not available, route to web_search
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", [])}

    # Try several possible retriever methods in order of likelihood
    try:
        # Some custom retrievers / langgraph retriever support .invoke(...)
        docs = retriever.invoke(question)
        return {"documents": docs, "question": question, "chat_history": state.get("chat_history", [])}
    except Exception:
        pass

    try:
        # LangChain retrievers often expose get_relevant_documents
        docs = retriever.get_relevant_documents(question)
        return {"documents": docs, "question": question, "chat_history": state.get("chat_history", [])}
    except Exception:
        pass

    try:
        # FAISS-like retrievers sometimes have similarity_search
        docs = retriever.similarity_search(question, k=4)
        return {"documents": docs, "question": question, "chat_history": state.get("chat_history", [])}
    except Exception as e:
        # Ultimately return empty documents and keep going
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", []), "error": str(e)}

def web_search(state: dict):
    """
    Perform Tavily web search and return as a Document.
    Handles both .invoke and .run bindings.
    """
    st.session_state["route"] = "web_search"
    question = state["question"]
    try:
        # prefer invoke if present
        results = None
        try:
            results = web_search_tool.invoke({"query": question})
        except Exception:
            try:
                results = web_search_tool.run(question)
            except Exception:
                results = web_search_tool({"query": question})  # some wrappers
        # results may be list of dicts with 'content' key or a simple string
        if isinstance(results, str):
            web_results = results
        elif isinstance(results, list):
            web_results = "\n".join([r.get("content", str(r)) for r in results])
        elif isinstance(results, dict):
            web_results = results.get("content", str(results))
        else:
            # fallback to stringification
            web_results = str(results)

        web_doc = Document(page_content=web_results)
        return {"documents": [web_doc], "question": question, "chat_history": state.get("chat_history", [])}
    except Exception as e:
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", []), "error": str(e)}

def chat_node(state):
    """
    Handle plain chat mode: directly ask LLM using chat history + question.
    This node produces a final generation and goes to END (no further generate node).
    """
    st.session_state["route"] = "chat"
    question = state["question"]
    chat_history = state.get("chat_history", [])

    # Build history-aware conversation (last 10 turns)
    messages = []
    for m in chat_history[-10:]:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        else:
            messages.append({"role": "assistant", "content": m["content"]})
    # Add current question
    messages.append({"role": "user", "content": question})

    # Call the Gemini chat model properly
    try:
        response = llm.invoke(messages)
        # response can be an AIMessage with .content or a string
        if hasattr(response, "content"):
            answer = response.content
        elif isinstance(response, str):
            answer = response
        else:
            answer = str(response)
    except Exception as e:
        answer = f"⚠️ Error invoking LLM for chat node: {e}"

    return {
        "documents": [],
        "question": question,
        "generation": answer,
        "chat_history": chat_history,
    }


def generate(state: dict):
    """
    Use retrieved documents (or web results) and chat history to create the final answer.
    This node supports RAG generation.
    """
    question = state["question"]
    documents = state.get("documents", []) or []
    chat_history = state.get("chat_history", []) or []

    # Build history-aware query
    history_context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-10:]])
    combined_question = f"Conversation so far:\n{history_context}\n\nUser's new question: {question}"

    # If we have a rag_chain configured, use it
    if rag_chain:
        try:
            # rag_chain expects context and question keys
            generation = rag_chain.invoke({"context": documents, "question": combined_question})
            # generation often is a string already
            answer = generation if isinstance(generation, str) else str(generation)
        except Exception:
            # fallback: call llm directly with assembled prompt
            ctx_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in documents]) if documents else ""
            prompt_input = f"Context:\n{ctx_text}\n\nQuestion:\n{combined_question}\n\nAnswer:"
            try:
                resp = llm.invoke({"input": prompt_input})
                if isinstance(resp, str):
                    answer = resp
                elif isinstance(resp, dict):
                    answer = resp.get("output_text") or resp.get("text") or str(resp)
                elif hasattr(resp, "output_text"):
                    answer = getattr(resp, "output_text")
                elif hasattr(resp, "content"):
                    answer = getattr(resp, "content")
                else:
                    answer = str(resp)
            except Exception as e:
                answer = "⚠️ Error generating answer: " + str(e)
    else:
        # No rag_chain; call LLM directly using context and question
        ctx_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in documents]) if documents else ""
        prompt_input = f"Context:\n{ctx_text}\n\nQuestion:\n{combined_question}\n\nAnswer:"
        try:
            resp = llm.invoke({"input": prompt_input})
            if isinstance(resp, str):
                answer = resp
            elif isinstance(resp, dict):
                answer = resp.get("output_text") or resp.get("text") or str(resp)
            elif hasattr(resp, "output_text"):
                answer = getattr(resp, "output_text")
            elif hasattr(resp, "content"):
                answer = getattr(resp, "content")
            else:
                answer = str(resp)
        except Exception as e:
            answer = "⚠️ Error generating answer: " + str(e)

    return {"documents": documents, "question": question, "generation": answer, "chat_history": chat_history}

def route_question(state: dict):
    """
    Use the structured LLM router to pick one of: vectorstore, web_search, chat.
    If vectorstore chosen but no retriever available, fallback to web_search.
    """
    question = state["question"]
    try:
        source = question_router.invoke({"question": question})
        ds = source.datasource
    except Exception:
        # fallback heuristic: simple keywords -> vectorstore if user mentions "report" "document" "pdf" "table"
        ql = question.lower()
        if any(k in ql for k in ["pdf", "document", "report", "paper", "section", "table"]):
            ds = "vectorstore"
        elif any(k in ql for k in ["who", "what", "when", "where", "why", "how", "latest", "current", "today"]):
            ds = "web_search"
        else:
            ds = "chat"

    # If user asked vectorstore but no retriever exists, fallback
    if ds == "vectorstore" and st.session_state.get("retriever") is None:
        ds = "web_search"
    return ds

# ------------------- Graph Build -------------------
workflow = StateGraph(GraphState)
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("chat", chat_node)

# Conditional start -> route_question resolves to 'web_search' or 'retrieve' or 'chat'
workflow.add_conditional_edges(START, route_question, {
    "web_search": "web_search",
    "vectorstore": "retrieve",
    "chat": "chat"
})

# Flows:
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
# chat node directly ends (it already returns generation)
workflow.add_edge("chat", END)

app = workflow.compile()

# ------------------- Chat UI Area -------------------
# Show previous chat in main panel
st.subheader("Conversation")
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'><b>Assistant:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Input box for new user message (supports multi-turn)
user_question = st.chat_input("Type your message...")

if user_question:
    # Append and display user message immediately
    st.session_state["chat_history"].append({"role": "user", "content": user_question})
    st.markdown(f"<div class='user-bubble'><b>You:</b> {user_question}</div>", unsafe_allow_html=True)

    # If user explicitly selected Normal Chat only, force chat node
    if mode == "Normal Chat only":
        chosen_start = "chat"
    else:
        chosen_start = None  # let graph decide

    with st.spinner("Processing..."):
        # invoke graph; pass chat_history to state
        # When chosen_start provided, we can directly call the specific node behavior:
        if chosen_start == "chat":
            # directly call chat_node
            result = chat_node({"question": user_question, "chat_history": st.session_state["chat_history"]})
        else:
            # Normal agentic flow
            # Graph start will call route_question which may pick chat/web/retrieve
            result = app.invoke({"question": user_question, "chat_history": st.session_state["chat_history"]})

    # Extract generation robustly
    generation = None
    if isinstance(result, dict):
        # chat node returns under 'generation'
        generation = result.get("generation") or result.get("result") or result.get("answer")
    else:
        generation = str(result)

    if generation:
        st.session_state["chat_history"].append({"role": "assistant", "content": generation})
        st.markdown(f"<div class='assistant-bubble'><b>Assistant:</b> {generation}</div>", unsafe_allow_html=True)

        # Show routing info
        if st.session_state.get("route"):
            st.markdown(f"<div class='meta'>Routed to: **{st.session_state['route']}**</div>", unsafe_allow_html=True)

        # Show short context documents if present
        docs = result.get("documents", []) if isinstance(result, dict) else []
        if docs:
            with st.expander("Context documents used (snippet)"):
                for d in docs:
                    content = getattr(d, "page_content", str(d))
                    st.write(content[:400] + ("..." if len(content) > 400 else ""))
    else:
        st.warning("No answer generated. Try rephrasing your question or check your API keys.")

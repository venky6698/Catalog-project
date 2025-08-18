# WebSocket RAG Chatbot – Full Stack Blueprint

This is a production‑grade, interview‑ready blueprint for a **WebSocket-enabled** chatbot that uses **MCP-style modular components**, **RAG (Retrieval-Augmented Generation)**, **LangChain + LangGraph** for agentic behavior, an **open-source LLM (Ollama / Llama 3 / Mistral)**, and a **React.js + shadcn/ui** front end with streaming tokens, citations, and tool-calls.

---

## 1) High-level Architecture

```
          ┌────────────────────────────────────────────────────────────────────┐
          │                              Frontend                              │
          │   React + shadcn/ui + Vercel AI SDK patterns + WebSocket client    │
          │    • Live token stream  • File upload (PDF)  • Citations view      │
          └───────────────▲───────────────────────────────────┬────────────────┘
                          │                                   │
                     WebSocket (JSON)                         │ HTTP (REST)
                          │                                   │
┌─────────────────────────┴──────────────────────────┐   ┌────▼─────────────────────┐
│                     FastAPI App                    │   │        /upload_pdf       │
│               /ws  (WebSocket endpoint)            │   │  PDF → chunk → embed     │
│   • Session state (chat history, tool events)       │   │  → Vector DB (FAISS)     │
│   • Streams tokens/citations/tool-calls             │   └──────────────────────────┘
│                                                     │
│   LangGraph Agent (MCP-style)                       │
│   ┌───────────────────────────────────────────────┐ │
│   │  Nodes: retrieve → route → generate → tools   │ │
│   │   • retriever (FAISS/Chroma)                  │ │
│   │   • generator (Ollama open-source LLM)        │ │
│   │   • tools: catalog_search, web_search, math   │ │
│   └───────────────────────────────────────────────┘ │
│                                                     │
│   Vector Store: FAISS + SentenceTransformers        │
│   PDF parsing: PyPDF + RecursiveCharacterSplitter   │
└─────────────────────────────────────────────────────┘
```

**Why this fits the prompt**

- **MCP-style** = modular, composable graph with explicit tool nodes.
- **RAG** = vector store retriever + context stuffing with citations.
- **Open-source LLM** = Ollama (local) running `llama3` or `mistral` with streaming.
- **Agentic** = LangGraph orchestrates retrieval ↔ tool-calls ↔ generation.
- **UI** = shadcn/ui components + WebSocket streaming; citation pane; tool call events.

---

## 2) Backend (FastAPI + LangChain + LangGraph + Ollama)

> **Dependencies**
>
> ```bash
> pip install fastapi uvicorn[standard] pydantic
> pip install langchain langchain-community langgraph
> pip install sentence-transformers faiss-cpu pypdf
> pip install sse-starlette  # optional
> # For local open-source LLM via Ollama
> # Install Ollama from https://ollama.com and pull a model, e.g.:
> #   ollama pull llama3  # or mistral
> pip install langchain-ollama  # if available; else use langchain-community ChatOllama
> ```

> **.env (example)**
>
> ```dotenv
> OLLAMA_MODEL=llama3
> VECTOR_DB_PATH=./data/faiss_index
> UPLOAD_DIR=./uploads
> ```

### 2.1 FastAPI app skeleton

```python
# app/main.py
import os
import json
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama  # open-source local via Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

from langgraph.graph import END, StateGraph
from langgraph.types import State

# ------------------ Settings ------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/faiss_index")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Vector Store ----------------
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize or create FAISS index
try:
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception:
    vectorstore = FAISS.from_texts(["Welcome! Upload PDFs to start."], embedding=embeddings, metadatas=[{"source": "seed"}])
    vectorstore.save_local(VECTOR_DB_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------- LLM ----------------------
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.2, streaming=True)

# --------------- Prompt w/ citations ----------
SYSTEM_PROMPT = (
    "You are a helpful catalog QA assistant. Use retrieved CONTEXT to answer. "
    "Cite sources as [#] where # is the index in the provided sources. "
    "If you don't know, say so briefly."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nCONTEXT:\n{context}\n\nReturn a concise answer with inline citations like [1], [2].")
])

# --------------- PDF Upload API ---------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Persist w/ metadata for citations
    vectorstore.add_documents(chunks)
    vectorstore.save_local(VECTOR_DB_PATH)

    return {"message": "PDF processed and indexed", "chunks": len(chunks)}

# --------------- Tooling (MCP-like) -----------
class ToolResult(BaseModel):
    name: str
    output: Any

async def tool_catalog_search(query: str) -> ToolResult:
    """Lightweight tool that searches the vector DB directly for catalog facts."""
    docs = retriever.get_relevant_documents(query)
    items = [{"page": d.metadata.get("page", None), "source": d.metadata.get("source", "pdf"), "text": d.page_content[:240]} for d in docs]
    return ToolResult(name="catalog_search", output=items)

async def tool_math(expr: str) -> ToolResult:
    try:
        val = eval(expr, {"__builtins__": {}}, {})  # sandboxed eval (still use carefully)
    except Exception as e:
        val = str(e)
    return ToolResult(name="math", output=val)

TOOLS = {
    "catalog_search": tool_catalog_search,
    "math": tool_math,
}

# ------------- LangGraph State/Nodes ----------
class ChatState(State):
    question: str
    context: str
    sources: List[Dict[str, Any]]
    answer: str
    tool_events: List[Dict[str, Any]]

async def retrieve_node(state: ChatState) -> ChatState:
    docs = retriever.get_relevant_documents(state["question"])
    ctx_lines = []
    sources = []
    for i, d in enumerate(docs, start=1):
        ctx_lines.append(f"[{i}] {d.page_content}")
        sources.append({
            "index": i,
            "source": d.metadata.get("source", "pdf"),
            "page": d.metadata.get("page"),
            "snippet": d.page_content[:220]
        })
    state["context"] = "\n\n".join(ctx_lines) if ctx_lines else ""
    state["sources"] = sources
    return state

async def route_node(state: ChatState) -> str:
    q = state["question"].lower()
    if q.startswith("calc ") or q.startswith("math "):
        return "tools"
    return "generate"

class WSStreamHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
    async def on_llm_new_token(self, token: str, **kwargs):
        await self.ws.send_json({"type": "token", "text": token})

async def generate_node(state: ChatState, websocket: WebSocket) -> ChatState:
    # Build chain: prompt(context) → LLM(stream)
    chain = PROMPT | llm
    cb = WSStreamHandler(websocket)
    resp = await chain.ainvoke({
        "question": state["question"],
        "context": state.get("context", "")
    }, config={"callbacks": [cb]})
    state["answer"] = resp.content
    return state

async def tools_node(state: ChatState, websocket: WebSocket) -> ChatState:
    q = state["question"].split(maxsplit=1)
    tool_call = q[0]
    arg = q[1] if len(q) > 1 else ""
    name = "catalog_search" if "catalog" in tool_call else "math"

    await websocket.send_json({"type": "tool_call", "name": name, "input": arg})
    result = await TOOLS[name](arg)
    await websocket.send_json({"type": "tool_result", "name": result.name, "output": result.output})

    # Use tool output as context for LLM answer
    state["context"] = json.dumps(result.output, ensure_ascii=False)

    return await generate_node(state, websocket)

# Build graph
workflow = StateGraph(ChatState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", lambda s, ws=None: generate_node(s, ws))
workflow.add_node("tools", lambda s, ws=None: tools_node(s, ws))
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "route")  # logical marker
workflow.add_conditional_edges("route", route_node, {"generate": "generate", "tools": "tools"})
workflow.add_edge("generate", END)
workflow.add_edge("tools", END)
app_state_machine = workflow.compile()

# --------------- WebSocket endpoint -----------
@app.websocket("/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            mtype = data.get("type")
            if mtype == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if mtype == "user_message":
                question = data.get("text", "").strip()
                if not question:
                    await ws.send_json({"type": "error", "message": "Empty question"})
                    continue

                # Run graph: retrieve → route → (generate|tools)
                init: ChatState = {
                    "question": question,
                    "context": "",
                    "sources": [],
      
```

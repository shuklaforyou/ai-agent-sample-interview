# 🤖 AI Agent Interview Code Samples

Clean, minimal, interview-ready Python code. One topic per file.  
All files use **Gemini free model** (`gemini-2.5-flash`) via `GOOGLE_API_KEY`.

---

## 🧠 Agent Patterns — Quick Reference

| # | Pattern | How it works | When to use |
|---|---------|-------------|-------------|
| 1 | **Single Node** | One LLM node, straight to END | Simplest graph, one task |
| 2 | **ReAct Loop** | Agent ↔ Tool loop via conditional edge | Tool-using agents |
| 3 | **Supervisor** | LLM classifies → routes to correct worker | Task routing |
| 4 | **Sequential** | node1 → node2 → node3 → END | Step-by-step pipeline |
| 5 | **Hierarchical** | Supervisor → Sub-supervisor → Worker (2 levels) | Large multi-domain systems |
| 6 | **Hub-and-Spoke** | Every agent passes through a central hub | Centralized logging/control |
| 7 | **Parallel** | All agents run on same input → merge results | Independent analysis tasks |
| 8 | **Reflection Loop** | Generate → Critique → loop until score ≥ threshold | Self-improving output |

---

## 📁 Folder Structure

```
Ai agent/
├── langchain_agents/
│   ├── 01_simple_agent.py              ← Single ReAct agent with one tool
│   ├── 02_agent_with_tools.py          ← Agent with multiple tools
│   ├── 03_array_of_agents.py           ← Sequential pipeline of agents
│   └── 04_supervisor_agent.py          ← Supervisor routes to worker agents
│
├── langgraph_agents/
│   ├── 05_langgraph_simple_agent.py    ← Minimal single-node StateGraph
│   ├── 06_langgraph_agent_with_tools.py← ReAct loop with conditional edges
│   ├── 07_langgraph_supervisor.py      ← Multi-agent graph with routing
│   ├── 08_langgraph_sequential.py      ← Step-by-step pipeline (node→node→node)
│   ├── 09_langgraph_hierarchical.py    ← 2-level supervisor hierarchy
│   ├── 10_langgraph_hub_spoke.py       ← Hub-and-spoke (single point of access)
│   ├── 11_langgraph_parallel.py        ← Fan-out to multiple agents, merge results
│   └── 12_langgraph_reflection.py      ← Generate → Critique → Refine loop
│
├── strands_agents/
│   ├── 01_strands_basic_agent.py       ← Basic agent with one tool
│   ├── 02_strands_multi_tool_agent.py  ← Agent picks from multiple tools
│   ├── 03_strands_system_prompt.py     ← Agent with role/system_instruction
│   ├── 04_strands_streaming.py         ← Token-by-token streaming output
│   ├── 05_strands_multi_turn.py        ← Multi-turn memory via start_chat()
│   └── 06_strands_supervisor_agent.py  ← Supervisor delegates to sub-agents
│
└── rag/
    ├── 08_rag_chunking_fixed_size.py   ← CharacterTextSplitter (chunk_size)
    ├── 09_rag_chunking_recursive.py    ← RecursiveCharacterTextSplitter ✅ most used
    ├── 10_rag_chunking_semantic.py     ← SemanticChunker (embedding-based)
    ├── 11_rag_retrieval_basic.py       ← FAISS vector store + top-k search
    ├── 12_rag_retrieval_chain.py       ← Full RAG chain using LCEL (pipe |)
    └── 13_rag_retrieval_mmr.py         ← MMR retrieval (diverse results)
```

---

## ⚡ Quick Install

```bash
# LangChain + LangGraph + RAG
pip install langchain langchain-google-genai langchain-community langchain-experimental langgraph faiss-cpu

# Gemini (Strands / direct usage)
pip install google-generativeai

# Set your free Gemini API key (get it from https://aistudio.google.com)
export GOOGLE_API_KEY="your-key-here"
```

---

## 🗂️ File-by-File Summary

### LangChain Agents (`langchain_agents/`)

| File | Pattern | Key Classes |
|------|---------|-------------|
| `01_simple_agent.py` | Single ReAct agent | `create_react_agent`, `AgentExecutor` |
| `02_agent_with_tools.py` | Multi-tool agent | `create_tool_calling_agent`, `@tool` |
| `03_array_of_agents.py` | Sequential agent pipeline | LCEL `\|` operator |
| `04_supervisor_agent.py` | Supervisor + workers | Routing via LLM output |

### LangGraph Agents (`langgraph_agents/`)

| File | Pattern | Key Concept |
|------|---------|-------------|
| `05_langgraph_simple_agent.py` | Single node | `StateGraph`, `TypedDict` |
| `06_langgraph_agent_with_tools.py` | ReAct tool loop | `ToolNode`, conditional edges |
| `07_langgraph_supervisor.py` | Supervisor routing | `add_conditional_edges`, workers |
| `08_langgraph_sequential.py` | Step-by-step pipeline | Fixed `add_edge` chain |
| `09_langgraph_hierarchical.py` | 2-level hierarchy | Supervisor → Sub-supervisor → Worker |
| `10_langgraph_hub_spoke.py` | Hub-and-spoke | All agents return to hub; hub controls routing |
| `11_langgraph_parallel.py` | Parallel fan-out/fan-in | Multiple agents on same input → merge |
| `12_langgraph_reflection.py` | Reflection loop | Generate → Critique → loop until score ≥ 8 |

### Strands / Gemini Agents (`strands_agents/`)

| File | Pattern | Key Concept |
|------|---------|-------------|
| `01_strands_basic_agent.py` | One tool + LLM | `tools=[]`, `start_chat()` |
| `02_strands_multi_tool_agent.py` | Multiple tools | Gemini picks the right tool |
| `03_strands_system_prompt.py` | Role/persona | `system_instruction=` |
| `04_strands_streaming.py` | Streaming output | `stream=True`, `flush=True` |
| `05_strands_multi_turn.py` | Conversation memory | `start_chat()` keeps history |
| `06_strands_supervisor_agent.py` | Supervisor pattern | Sub-agents wrapped as functions |

### RAG — Chunking (`rag/`)

| File | Strategy | Key Class |
|------|----------|-----------|
| `08_rag_chunking_fixed_size.py` | Fixed char count | `CharacterTextSplitter` |
| `09_rag_chunking_recursive.py` | `\n\n` → `\n` → word | `RecursiveCharacterTextSplitter` |
| `10_rag_chunking_semantic.py` | Embedding similarity | `SemanticChunker` |

### RAG — Retrieval (`rag/`)

| File | Strategy | Key Concept |
|------|----------|-------------|
| `11_rag_retrieval_basic.py` | Top-k similarity | `FAISS.as_retriever()` |
| `12_rag_retrieval_chain.py` | Full RAG LCEL chain | `RunnablePassthrough`, pipe `\|` |
| `13_rag_retrieval_mmr.py` | Diverse results | `search_type="mmr"`, `lambda_mult` |

---

## 🔑 Key Concepts to Remember in Interviews

### Agents
- **ReAct** = Reason + Act loop (think → use tool → observe → repeat)
- **Supervisor** = LLM classifies task → routes to the correct worker agent
- **Graph (LangGraph)** = nodes are functions, edges are transitions, state is a shared `TypedDict`
- **Strands / Gemini** = `GenerativeModel(tools=[...])` + `start_chat(enable_automatic_function_calling=True)`

### RAG — Chunking
| Term | Meaning | Example |
|------|---------|---------|
| `chunk_size` | Max characters per chunk | `150` chars |
| `chunk_overlap` | Shared text between chunks | `30` chars overlap to preserve context |
| **Fixed** | Split every N chars | Simple, fast, may cut sentences |
| **Recursive** | Try `\n\n` → `\n` → space | Keeps paragraphs intact ✅ most common |
| **Semantic** | Split where meaning changes | Uses embeddings to detect topic shifts |

### RAG — Retrieval
| Term | Meaning |
|------|---------|
| **Embeddings** | Text → numbers for similarity search |
| **FAISS** | Fast in-memory vector store by Meta |
| **Top-k** | Return the k most similar chunks |
| **MMR** | Balance relevance + diversity (avoids near-duplicate chunks) |
| **LCEL pipe `\|`** | `retriever \| prompt \| llm \| parser` — clean chain syntax |

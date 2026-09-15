"""
09_langgraph_hierarchical.py
-----------------------------
Pattern: Hierarchical Graph (2-level Supervisor → Sub-supervisor → Workers)
What:    Top supervisor routes to a domain sub-supervisor; sub-supervisor routes to workers
Meaning: Scales multi-agent systems — each domain manages its own specialists
Example: task → top_supervisor → "tech" → tech_supervisor → "coder" or "debugger"

Level 1:  top_supervisor         (routes: tech | business)
Level 2:  tech_supervisor        (routes: coder | debugger)
          business_supervisor    (routes: analyst | writer)
Level 3:  coder / debugger / analyst / writer  (workers)
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import TypedDict

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    task: str
    domain: str       # "tech" or "business"
    sub_route: str    # "coder" / "debugger" / "analyst" / "writer"
    result: str

# ── Level 1: Top Supervisor ────────────────────────────────────────────────────
def top_supervisor(state: State) -> State:
    resp = llm.invoke([HumanMessage(content=
        f"Classify as 'tech' or 'business' (one word): {state['task']}"
    )]).content.strip().lower()
    return {"domain": resp}

# ── Level 2: Tech Sub-supervisor ──────────────────────────────────────────────
def tech_supervisor(state: State) -> State:
    resp = llm.invoke([HumanMessage(content=
        f"Classify as 'coder' or 'debugger' (one word): {state['task']}"
    )]).content.strip().lower()
    return {"sub_route": resp}

# ── Level 2: Business Sub-supervisor ──────────────────────────────────────────
def business_supervisor(state: State) -> State:
    resp = llm.invoke([HumanMessage(content=
        f"Classify as 'analyst' or 'writer' (one word): {state['task']}"
    )]).content.strip().lower()
    return {"sub_route": resp}

# ── Level 3: Workers ──────────────────────────────────────────────────────────
def coder(state: State) -> State:
    result = llm.invoke([HumanMessage(content=f"Write Python code for: {state['task']}")]).content
    return {"result": result}

def debugger(state: State) -> State:
    result = llm.invoke([HumanMessage(content=f"Debug this issue: {state['task']}")]).content
    return {"result": result}

def analyst(state: State) -> State:
    result = llm.invoke([HumanMessage(content=f"Analyze this business problem: {state['task']}")]).content
    return {"result": result}

def writer(state: State) -> State:
    result = llm.invoke([HumanMessage(content=f"Write a business report for: {state['task']}")]).content
    return {"result": result}

# ── Routing functions ──────────────────────────────────────────────────────────
def route_domain(state: State) -> str:
    return state["domain"]       # → "tech" or "business"

def route_sub(state: State) -> str:
    return state["sub_route"]    # → "coder" / "debugger" / "analyst" / "writer"

# ── Build graph ────────────────────────────────────────────────────────────────
graph = StateGraph(State)

# Add all nodes
graph.add_node("top_supervisor",      top_supervisor)
graph.add_node("tech_supervisor",     tech_supervisor)
graph.add_node("business_supervisor", business_supervisor)
graph.add_node("coder",    coder)
graph.add_node("debugger", debugger)
graph.add_node("analyst",  analyst)
graph.add_node("writer",   writer)

# Level 1 → Level 2
graph.set_entry_point("top_supervisor")
graph.add_conditional_edges("top_supervisor", route_domain, {
    "tech":     "tech_supervisor",
    "business": "business_supervisor",
})

# Level 2 → Level 3
graph.add_conditional_edges("tech_supervisor", route_sub, {
    "coder":    "coder",
    "debugger": "debugger",
})
graph.add_conditional_edges("business_supervisor", route_sub, {
    "analyst": "analyst",
    "writer":  "writer",
})

# Workers → END
for worker in ["coder", "debugger", "analyst", "writer"]:
    graph.add_edge(worker, END)

app = graph.compile()

if __name__ == "__main__":
    tasks = [
        "Write a Python function to sort a list.",
        "Analyse quarterly sales performance.",
    ]
    for task in tasks:
        result = app.invoke({"task": task, "domain": "", "sub_route": "", "result": ""})
        print(f"\nTask:      {task}")
        print(f"Domain:    {result['domain']}  →  {result['sub_route']}")
        print(f"Result:    {result['result'][:200]}...")

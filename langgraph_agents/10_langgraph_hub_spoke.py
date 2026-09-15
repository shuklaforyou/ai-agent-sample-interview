"""
10_langgraph_hub_spoke.py
--------------------------
Pattern: Hub-and-Spoke (Single Point of Access)
What:    Every agent goes through a central HUB node before and after its work
Meaning: Hub logs, validates, or enriches state at every step — single control point
Example: hub → coder → hub → hub decides: done or send to next spoke

Flow:
  hub → pick spoke → spoke runs → hub → pick next or END
  (hub is always in the middle — nothing bypasses it)
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
    next_spoke: str    # hub sets this to decide where to go
    draft: str         # intermediate content
    final: str         # finished content
    step: int          # tracks how many spokes have run

# ── Hub: central controller ────────────────────────────────────────────────────
def hub(state: State) -> State:
    step = state.get("step", 0)
    # Decide which spoke to use based on step number
    if step == 0:
        next_spoke = "researcher"
    elif step == 1:
        next_spoke = "writer"
    elif step == 2:
        next_spoke = "reviewer"
    else:
        next_spoke = "done"
    print(f"[HUB] step={step} → next={next_spoke}")
    return {"next_spoke": next_spoke, "step": step + 1}

# ── Spoke 1: Researcher ────────────────────────────────────────────────────────
def researcher(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Give 3 key facts about: {state['task']}"
    )]).content
    return {"draft": result}

# ── Spoke 2: Writer ───────────────────────────────────────────────────────────
def writer(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Write a short paragraph using these facts:\n{state['draft']}"
    )]).content
    return {"draft": result}

# ── Spoke 3: Reviewer ─────────────────────────────────────────────────────────
def reviewer(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Polish and finalize this paragraph:\n{state['draft']}"
    )]).content
    return {"final": result}

# ── Hub routing ───────────────────────────────────────────────────────────────
def route_from_hub(state: State) -> str:
    return state["next_spoke"]   # returns "researcher", "writer", "reviewer", or "done"

# ── Build graph ────────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("hub",        hub)
graph.add_node("researcher", researcher)
graph.add_node("writer",     writer)
graph.add_node("reviewer",   reviewer)

# Hub is entry point and routes to each spoke
graph.set_entry_point("hub")
graph.add_conditional_edges("hub", route_from_hub, {
    "researcher": "researcher",
    "writer":     "writer",
    "reviewer":   "reviewer",
    "done":       END,           # hub decides when we are finished
})

# Every spoke returns to hub (single point of access)
graph.add_edge("researcher", "hub")
graph.add_edge("writer",     "hub")
graph.add_edge("reviewer",   "hub")

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "task": "Climate change effects on agriculture",
        "next_spoke": "", "draft": "", "final": "", "step": 0
    })
    print(f"\nFinal output:\n{result['final']}")

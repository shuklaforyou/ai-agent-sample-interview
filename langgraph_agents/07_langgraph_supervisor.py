"""
07_langgraph_supervisor.py
---------------------------
Topic: LangGraph Supervisor — multi-agent graph with routing (Gemini)
What:    Supervisor node classifies the task and routes to coder or writer node
Meaning: conditional_edges map the supervisor's output to the correct worker node
Example: "Write a poem" → supervisor says "writer" → writer node handles it
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    user_input: str
    route: str
    result: str

# ── Node 1: Supervisor decides route ──────────────────────────────────────────
def supervisor(state: State) -> State:
    prompt = f"""
    You are a supervisor. Classify the task as one word: 'coder' or 'writer'.
    Task: {state['user_input']}
    Answer (one word):
    """
    route = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    return {"route": route}

# ── Node 2: Coder worker ───────────────────────────────────────────────────────
def coder(state: State) -> State:
    result = llm.invoke([
        SystemMessage(content="You are a Python expert."),
        HumanMessage(content=state["user_input"])
    ]).content
    return {"result": result}

# ── Node 3: Writer worker ──────────────────────────────────────────────────────
def writer(state: State) -> State:
    result = llm.invoke([
        SystemMessage(content="You are a creative writer."),
        HumanMessage(content=state["user_input"])
    ]).content
    return {"result": result}

# ── Routing function ───────────────────────────────────────────────────────────
def route_to_worker(state: State) -> str:
    return state["route"]   # returns "coder" or "writer"

# ── Build graph ────────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("coder", coder)
graph.add_node("writer", writer)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_worker, {
    "coder": "coder",
    "writer": "writer",
})
graph.add_edge("coder", END)
graph.add_edge("writer", END)

app = graph.compile()

if __name__ == "__main__":
    tasks = [
        "Write a Python function to check if a number is prime.",
        "Write a short poem about the ocean.",
    ]
    for task in tasks:
        result = app.invoke({"user_input": task, "route": "", "result": ""})
        print(f"\nTask:   {task}")
        print(f"Route:  {result['route']}")
        print(f"Result: {result['result']}")

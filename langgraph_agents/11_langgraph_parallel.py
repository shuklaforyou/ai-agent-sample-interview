"""
11_langgraph_parallel.py
-------------------------
Pattern: Parallel Fan-out / Fan-in (multiple agents run on same input, results merged)
What:    One input is sent to N agents simultaneously; their outputs are merged into one
Meaning: Each agent works independently — no waiting — then results are combined
Example: topic → [fact_finder, critic, summarizer] all run → merger combines all 3

Note: LangGraph does not support true async fan-out natively in basic StateGraph.
      The common interview pattern is to invoke all branches in one "fan_out" node
      and merge in a "merge" node — this represents the parallel pattern clearly.
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
    topic: str
    facts: str       # output from fact_finder agent
    critique: str    # output from critic agent
    summary: str     # output from summarizer agent
    merged: str      # combined final answer

# ── Fan-out: all 3 agents run in ONE node (simulates parallel execution) ───────
def fan_out(state: State) -> State:
    topic = state["topic"]
    facts     = llm.invoke([HumanMessage(content=f"Give 3 key facts about: {topic}")]).content
    critique  = llm.invoke([HumanMessage(content=f"What are 2 criticisms of: {topic}")]).content
    summary   = llm.invoke([HumanMessage(content=f"Summarize in 1 sentence: {topic}")]).content
    return {"facts": facts, "critique": critique, "summary": summary}

# ── Fan-in / Merge: combines all outputs into one ─────────────────────────────
def merge(state: State) -> State:
    merged = llm.invoke([HumanMessage(content=f"""
Combine the following three perspectives into one balanced paragraph:

FACTS: {state['facts']}

CRITIQUE: {state['critique']}

SUMMARY: {state['summary']}
""")]).content
    return {"merged": merged}

# ── Build graph ────────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("fan_out", fan_out)   # splits work (runs all agents)
graph.add_node("merge",   merge)     # joins results

graph.set_entry_point("fan_out")
graph.add_edge("fan_out", "merge")
graph.add_edge("merge", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "topic": "Artificial Intelligence in healthcare",
        "facts": "", "critique": "", "summary": "", "merged": ""
    })
    print(f"Facts:\n{result['facts']}\n")
    print(f"Critique:\n{result['critique']}\n")
    print(f"Summary:\n{result['summary']}\n")
    print(f"Merged:\n{result['merged']}")

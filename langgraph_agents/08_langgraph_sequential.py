"""
08_langgraph_sequential.py
---------------------------
Pattern: Sequential Pipeline (step-by-step nodes in a graph)
What:    Nodes run one after another in a fixed linear order
Meaning: Each node reads state, does one job, writes result back, next node picks up
Example: input → summarize → translate → sentiment → END

Difference from 03 (LCEL chains):
  - LCEL: uses Python | pipe operator, no explicit state
  - LangGraph sequential: uses StateGraph with named nodes, shared state dict
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

# ── Shared state ───────────────────────────────────────────────────────────────
class State(TypedDict):
    original: str
    summary: str
    translation: str
    sentiment: str

# ── Step 1: Summarize ──────────────────────────────────────────────────────────
def summarize(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Summarize in 1 sentence:\n{state['original']}"
    )]).content
    return {"summary": result}

# ── Step 2: Translate ──────────────────────────────────────────────────────────
def translate(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Translate to Hindi:\n{state['summary']}"
    )]).content
    return {"translation": result}

# ── Step 3: Sentiment ──────────────────────────────────────────────────────────
def sentiment(state: State) -> State:
    result = llm.invoke([HumanMessage(
        content=f"Sentiment (positive/negative/neutral) of:\n{state['summary']}"
    )]).content
    return {"sentiment": result}

# ── Build linear graph ────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("summarize",  summarize)
graph.add_node("translate",  translate)
graph.add_node("sentiment",  sentiment)

graph.set_entry_point("summarize")
graph.add_edge("summarize", "translate")   # step 1 → step 2
graph.add_edge("translate", "sentiment")   # step 2 → step 3
graph.add_edge("sentiment", END)           # step 3 → done

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "original": "LangChain is a great framework for building AI agents quickly.",
        "summary": "", "translation": "", "sentiment": ""
    })
    print(f"Summary:     {result['summary']}")
    print(f"Translation: {result['translation']}")
    print(f"Sentiment:   {result['sentiment']}")

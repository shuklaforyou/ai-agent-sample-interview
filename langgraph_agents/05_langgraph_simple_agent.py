"""
05_langgraph_simple_agent.py
-----------------------------
Topic: Simple LangGraph Agent — single node graph (Gemini)
What:    One LLM node inside a StateGraph; messages flow in and out
Meaning: Graph = state machine; node = a function that transforms state
Example: [HumanMessage("Capital of Japan?")] → LLM node → [AIMessage("Tokyo")]
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

load_dotenv()

# ── State ─────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List[HumanMessage]

# ── LLM node ──────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

def call_llm(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# ── Graph ─────────────────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.set_entry_point("llm")
graph.add_edge("llm", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="What is the capital of Japan?")]
    })
    print(result["messages"][-1].content)

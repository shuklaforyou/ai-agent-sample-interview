"""
06_langgraph_agent_with_tools.py
---------------------------------
Topic: LangGraph Agent with Tool-Calling Loop — ReAct style (Gemini)
What:    Agent node calls LLM; if LLM picks a tool, ToolNode runs it; loop repeats
Meaning: Conditional edge decides: tool call → continue loop | no tool → END
Example: "8 × 9?" → agent calls multiply(8,9) → tool returns 72 → agent answers
"""

import os
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated

load_dotenv()

# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

tools = [multiply]

# ── LLM bound to tools ────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
).bind_tools(tools)

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, operator.add]

# ── Agent node ────────────────────────────────────────────────────────────────
def agent_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ── Routing: loop if tool call, else stop ─────────────────────────────────────
def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ── Graph ─────────────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)   # loop or end
graph.add_edge("tools", "agent")                        # tools → back to agent

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "messages": [HumanMessage(content="What is 8 multiplied by 9?")]
    })
    print(result["messages"][-1].content)

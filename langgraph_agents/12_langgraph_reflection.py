"""
12_langgraph_reflection.py
---------------------------
Pattern: Reflection / Self-Critique Loop
What:    Agent generates a draft → critic evaluates it → if not good enough, loop back
Meaning: Iterative self-improvement — the agent refines its own output automatically
Example: generate draft → critique → score < 8 → regenerate → critique → score ≥ 8 → END

Key LangGraph concept: conditional edge creates the feedback loop
  generate → critique → [loop back OR end] based on quality score
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

MAX_ITERATIONS = 3   # safety limit to avoid infinite loops

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    topic: str
    draft: str
    feedback: str
    score: int        # critic scores the draft 1-10
    iterations: int

# ── Node 1: Generator ─────────────────────────────────────────────────────────
def generate(state: State) -> State:
    feedback = state.get("feedback", "")
    prompt = f"Write a short paragraph about: {state['topic']}"
    if feedback:
        prompt += f"\n\nPrevious feedback to incorporate:\n{feedback}"
    draft = llm.invoke([HumanMessage(content=prompt)]).content
    return {"draft": draft, "iterations": state.get("iterations", 0) + 1}

# ── Node 2: Critic ────────────────────────────────────────────────────────────
def critique(state: State) -> State:
    response = llm.invoke([HumanMessage(content=f"""
Review this paragraph and respond in exactly this format:
SCORE: <number 1-10>
FEEDBACK: <one sentence of improvement>

Paragraph:
{state['draft']}
""")]).content

    # Parse score from response
    score = 5  # default
    feedback = response
    for line in response.splitlines():
        if line.startswith("SCORE:"):
            try:
                score = int(line.split(":")[1].strip())
            except ValueError:
                pass
        if line.startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()

    return {"score": score, "feedback": feedback}

# ── Routing: loop if score < 8 and under max iterations ───────────────────────
def should_refine(state: State) -> str:
    if state["score"] >= 8 or state.get("iterations", 0) >= MAX_ITERATIONS:
        return END      # good enough or hit limit
    return "generate"   # loop back for another attempt

# ── Build graph ────────────────────────────────────────────────────────────────
graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("critique", critique)

graph.set_entry_point("generate")
graph.add_edge("generate", "critique")                    # always critique after generate
graph.add_conditional_edges("critique", should_refine)    # loop or end

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "topic": "Benefits of daily exercise",
        "draft": "", "feedback": "", "score": 0, "iterations": 0
    })
    print(f"Iterations:  {result['iterations']}")
    print(f"Final score: {result['score']}/10")
    print(f"Final draft:\n{result['draft']}")

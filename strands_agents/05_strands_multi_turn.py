"""
Strands SDK – Multi-Turn Conversation (Memory)
-----------------------------------------------
Key concepts:
  - Agent maintains conversation history across calls
  - Each agent() call appends to the same message thread
  - No extra config needed — history is built-in
"""

from strands import Agent, tool


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def remember_fact(fact: str) -> str:
    """Acknowledge and store a fact given by the user."""
    return f"Got it! I'll remember: '{fact}'"


# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(tools=[remember_fact])

if __name__ == "__main__":
    # Turn 1: give a fact
    print(agent("Please remember that my favourite color is blue."))

    # Turn 2: recall from memory
    print(agent("What is my favourite color?"))

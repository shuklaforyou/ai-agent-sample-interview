"""
Strands SDK – Basic Agent with One Tool
---------------------------------------
Key concepts:
  - @tool decorator to define a tool
  - Agent() to create an agent
  - agent() to run it with a prompt
"""

import os
from dotenv import load_dotenv
from strands import Agent, tool

load_dotenv()


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(tools=[add_numbers])

if __name__ == "__main__":
    response = agent("What is 7 + 15?")
    print(response)

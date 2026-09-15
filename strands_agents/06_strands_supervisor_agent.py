"""
Strands SDK – Supervisor Agent Pattern
----------------------------------------
Architecture:
  - Two specialist sub-agents: math_agent, text_agent
  - One supervisor_agent that delegates tasks to them
  - Supervisor wraps each sub-agent as a @tool

Key concepts:
  - Agent-as-tool pattern (supervisor calls sub-agents)
  - Routing based on task type
  - Each agent has its own system_prompt for specialization
"""

from strands import Agent, tool


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Agent 1: Math Specialist
# ══════════════════════════════════════════════════════════════════════════════

@tool
def square(n: int) -> int:
    """Return the square of a number."""
    return n * n


@tool
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


math_agent = Agent(
    system_prompt="You are a math specialist. Only answer math questions.",
    tools=[square, subtract],
)


# ══════════════════════════════════════════════════════════════════════════════
# Sub-Agent 2: Text Specialist
# ══════════════════════════════════════════════════════════════════════════════

@tool
def word_count(text: str) -> int:
    """Count the number of words in the text."""
    return len(text.split())


@tool
def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


text_agent = Agent(
    system_prompt="You are a text processing specialist. Only handle text tasks.",
    tools=[word_count, to_uppercase],
)


# ══════════════════════════════════════════════════════════════════════════════
# Supervisor: wraps sub-agents as tools and routes tasks
# ══════════════════════════════════════════════════════════════════════════════

@tool
def call_math_agent(query: str) -> str:
    """Delegate a math question to the math specialist agent."""
    return str(math_agent(query))


@tool
def call_text_agent(query: str) -> str:
    """Delegate a text processing task to the text specialist agent."""
    return str(text_agent(query))


supervisor = Agent(
    system_prompt=(
        "You are a supervisor. Route math questions to call_math_agent "
        "and text tasks to call_text_agent. Never answer directly."
    ),
    tools=[call_math_agent, call_text_agent],
)


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(supervisor("What is the square of 12?"))
    print(supervisor("Convert 'hello world' to uppercase."))
    print(supervisor("How many words are in 'the quick brown fox'?"))

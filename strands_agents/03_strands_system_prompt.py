"""
Strands SDK – Agent with System Prompt
---------------------------------------
Key concepts:
  - system_prompt sets the agent's persona/role
  - Useful for specializing agent behavior
  - Combined with tools for a focused assistant
"""

from strands import Agent, tool


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a math tutor. You only answer math-related questions.
Always explain the steps briefly before giving the final answer.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(system_prompt=SYSTEM_PROMPT, tools=[multiply])

if __name__ == "__main__":
    response = agent("What is 8 multiplied by 9?")
    print(response)

"""
Strands SDK – Streaming Agent Response
----------------------------------------
Key concepts:
  - stream=True enables token-by-token streaming
  - Iterate over the stream to print as tokens arrive
  - Great for chat-like UX
"""

from strands import Agent, tool


# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def greet(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}! Welcome aboard."


# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(tools=[greet])

if __name__ == "__main__":
    # Stream the response token by token
    for chunk in agent.stream("Greet Alice warmly."):
        print(chunk, end="", flush=True)
    print()  # newline at the end

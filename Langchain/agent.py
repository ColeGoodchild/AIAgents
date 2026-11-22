
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

# Load API key from .env
load_dotenv()

# Directory the agent is allowed to access
WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


# TOOL 1: List files
@tool
def list_files() -> str:
    """List all files available in the workspace directory."""
    files = [
        file.name
        for file in WORKSPACE.iterdir()
        if file.is_file()
    ]
    return str(files)


# TOOL 2: Read files
@tool
def read_file(filename: str) -> str:
    """Read the contents of a text file in the workspace."""
    try:
        path = (WORKSPACE / filename).resolve()

        if not path.is_relative_to(WORKSPACE):
            return "Access denied."

        if not path.is_file():
            return "File not found."

        if path.suffix.lower() not in {".txt", ".md"}:
            return "Only .txt and .md files are supported."

        if path.stat().st_size > 100_000:
            return "File is too large."

        return path.read_text(encoding="utf-8")

    except (OSError, ValueError, UnicodeError) as e:
        return f"Error reading file: {e}"


# Create LLM
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# Native LangGraph short-term memory
memory = InMemorySaver()


# Create agent
agent = create_agent(
    model=model,
    tools=[list_files, read_file],
    system_prompt=(
        "You are a helpful AI assistant. "
        "You can list and read files in the workspace. "
        "Use tools when needed. "
        "Use conversation history to answer follow-up questions. "
        "Never invent file contents."
    ),
    checkpointer=memory
)


# Interactive CLI
print("AI Agent Started!")
print("Type 'exit' to quit.\n")

thread_id = "conversation_1"

while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_input
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    print("\nAgent:", result["messages"][-1].content)
    print()

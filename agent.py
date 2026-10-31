
from datetime import datetime
import subprocess

from ollama import chat

MODEL = "qwen3:4b"

# Only these tools are available to the model.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_calculator",
            "description": "Open the Windows Calculator application.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def get_local_time():
    return datetime.now().astimezone().strftime(
        "%A, %d %B %Y, %I:%M %p"
    )


def open_calculator():
    subprocess.Popen(["calc.exe"], shell=False)
    return "Windows Calculator was opened."


AVAILABLE_TOOLS = {
    "get_local_time": get_local_time,
    "open_calculator": open_calculator,
}

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful personal AI assistant running locally. "
            "Use available tools when they are useful. "
            "Never claim an action succeeded unless its tool succeeded. "
            "Do not pretend you have access to tools that are not provided. "
            "Ask permission before destructive or sensitive actions."
        ),
    }
]

print("Personal AI is ready. Type 'exit' to quit.")

while True:
    user_text = input("\nYou: ").strip()

    if user_text.lower() in {"exit", "quit"}:
        break

    if not user_text:
        continue

    messages.append({"role": "user", "content": user_text})

    for _ in range(5):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            options={"num_ctx": 8192},
        )

        message = response.message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            print(f"\nAssistant: {message.content}")
            break

        for call in message.tool_calls:
            name = call.function.name
            arguments = call.function.arguments

            if name not in AVAILABLE_TOOLS:
                result = f"Tool not allowed: {name}"
            else:
                try:
                    result = AVAILABLE_TOOLS[name](**arguments)
                except Exception as exc:
                    result = f"Tool failed: {type(exc).__name__}: {exc}"

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": str(result),
            })
    else:
        print("Assistant: I reached the tool-call limit for this request.")

from shared.llm_service import generate_response
from shared.tools import TOOLS
import json

def create_plan(query):
    system_prompt = """
You are an AI planning agent.

Your job is to break the user request into clear steps.

Return the plan as numbered steps.
Do not answer the question.
Only produce the plan.
"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": query}
    ]

    plan = generate_response(messages)

    return plan

def execute_plan(query, plan):

    system_prompt = f"""
You are an autonomous AI agent.

Follow the plan and solve the task.

You can use tools when necessary.

Available tools:
web_search(query)

If you want to use a tool return:

TOOL: tool_name
INPUT: tool_input

If you want to finish return:

FINAL: answer
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"PLAN:\n{plan}"}
    ]

    while True:

        response = generate_response(messages)

        if response.startswith("FINAL:"):
            return response.replace("FINAL:", "").strip()

        if response.startswith("TOOL:"):

            lines = response.split("\n")
            tool_name = lines[0].split(":")[1].strip()
            tool_input = lines[1].split(":")[1].strip()

            tool_result = TOOLS[tool_name](tool_input)

            messages.append({
                "role": "assistant",
                "content": response
            })

            messages.append({
                "role": "user",
                "content": f"TOOL RESULT: {tool_result}"
            })


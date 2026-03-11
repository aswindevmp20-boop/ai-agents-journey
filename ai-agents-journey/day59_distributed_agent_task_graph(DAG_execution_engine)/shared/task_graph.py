from shared.llm_service import generate_response
import json
from shared.tools import TOOLS
def generate_task_graph(query):
    system_prompt = """
You are an AI planner.

Break the task into 3 parallel research tasks.

Return JSON:
{
    "tasks":[
        "task1",
        "task2",
        "task3"
    ]
}
"""
    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": query}
    ]

    response = generate_response(messages)

    try:
        data= json.loads(response)
        return data["tasks"]
    except:
        return [query]
    

def execute_task_graph(tasks):
    results = []
    for task in tasks:
        result = TOOLS["web_search"](task)
        results.append(result)
    
    return results

def combine_results(query, results):
    combined = "\n\n".join(results)
    system_prompt = """
You are a research analyst.

Combine the information into a structured answer.
"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": f"Query: {query} \n\n Data: {combined}"}
    ]

    return generate_response(messages)

def run_dag_agent(query):
    tasks = generate_task_graph(query)
    results = execute_task_graph(tasks)
    final = combine_results(query, results)
    return final
from shared.llm_service import generate_response

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

    system_prompt="""
You are an AI worker agent.
Follow this plan step by step to answer the question.

PLAN : {plan}
"""

    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user", "content":query}
    ]

    result = generate_response(messages)

    return result
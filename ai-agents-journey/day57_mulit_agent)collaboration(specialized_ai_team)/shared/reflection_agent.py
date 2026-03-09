from shared.llm_service import generate_response

def critique_answer(query, answer):

    system_prompt = """
You are an AI Critic.

Evaluate the answer to the user question.
If the answer is good, respond with:
GOOD
If the answer is incompelete or unclear, respond with:
IMPORVE: followed by suggestions.

"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": f"Question: {query} \n Answer: {answer}"}
    ]

    critique = generate_response(messages)

    return critique

def refine_answer(query, answer, critique):
    system_prompt = """
You are an AI assistant improving a previous answer.

Use the critique to produce a better, clearer response.
"""

    messages = [
        {"role":"system", "content": system_prompt},
        {"role":"user", "content": f"""
Question: {query}

Previous Answer:
{answer}

Critique:
{critique}

provide an improved answer.
"""
}
    ]

    improved_answer = generate_response(messages)
    return improved_answer
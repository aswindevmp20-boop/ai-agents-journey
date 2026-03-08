START = "START"
PLAN = "PLAN"
RETRIEVE = "RETRIEVE"
ANALYZE = "ANALYZE"
FINALIZE = "FINALIZE"
END = "END"


from shared.planner_agent import create_plan
from shared.tools import TOOLS
from shared.llm_service import generate_response

def run_workflow(query):

    state = START
    plan = None
    retrieved_data = None

    while state != END:

        if state == START:
            state = PLAN
        
        elif state == PLAN:
            plan = create_plan(query)
            state = RETRIEVE

        elif state == RETRIEVE:

            retrieved_data = TOOLS["web_search"](query)
            state = ANALYZE
        
        elif state ==ANALYZE:
            messages = [
                {"role":"system", "content": "Analyze the retrieved data."},
                {"role":"user", "content": retrieved_data}
            ]

            analysis = generate_response(messages)
            retrieved_data = analysis
            state = FINALIZE

        elif state == FINALIZE:

            messages = [
                {"role":"system", "content": "Produce the final answer."},
                {"role":"user", "content": retrieved_data}
            ]

            final_answer = generate_response(messages)
            state = END

    return final_answer
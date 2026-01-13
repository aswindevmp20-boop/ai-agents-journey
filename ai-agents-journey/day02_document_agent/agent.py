import json
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Python tool to read files

def read_file(path):
    try:
        with open(path,"r", encoding = "utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "ERROR: File not found."


TOOL_FUNCTIONS = {
    "read_file": read_file,
}

# Declare Tool Schema

tools = [
    {
        "type":"function",
        "function":{
            "name":"read_file",
            "description":"Read text from a file and return its content.",
            "parameters":{
                "type":"object",
                "properties":{
                    "path":{"type":"string"}
                },
                "required":["path"]
            }
        }
    }
]

# User Task

query = (
    "Read the file './sample.txt' and summarize it in 4 bullet points."
)

response = client.chat.completions.create(
    model = "llama-3.3-70b-versatile",
    messages = [{"role":"user","content": query}],
    tools=tools
)

print("\nRAW RESPONSE:\n", response, "\n")

msg = response.choices[0].message

# Handle Tool call

if msg.tool_calls:
    for call in msg.tool_calls:
        fn_name = call.function.name
        raw_args = call.function.arguments

        print("Raw args:", raw_args)

        args = json.loads(raw_args)
        result = TOOL_FUNCTIONS[fn_name](**args)

        print("\nTool result (file content):\n", result)

        # Send Result back to the model

        followup = client.chat.completions.create(
            model = "llama-3.3-70b-versatile",
            messages=[
                {"role":"user", "content": query},
                {"role":"assistant", "tool_calls": msg.tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": call.id,   
                    "content": result
                }
            ]
        )

        print("\n FINAL SUMMARY: \n", followup.choices[0].message.content)





# Expected Output:-

# RAW RESPONSE:
#  ChatCompletion(id='chatcmpl-d9b5bcfa-28a5-4218-a470-028b7f34ede4', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content=None, role='assistant', annotations=None, executed_tools=None, function_call=None, reasoning=None, tool_calls=[ChatCompletionMessageToolCall(id='wxmq92hsf', function=Function(arguments='{"path":"./sample.txt"}', name='read_file'), type='function')]))], created=1768276300, model='llama-3.3-70b-versatile', object='chat.completion', mcp_list_tools=None, service_tier='on_demand', system_fingerprint='fp_c06d5113ec', usage=CompletionUsage(completion_tokens=15, prompt_tokens=234, total_tokens=249, completion_time=0.047285142, completion_tokens_details=None, prompt_time=0.011670936, prompt_tokens_details=None, queue_time=0.058025477, total_time=0.058956078), usage_breakdown=None, x_groq=XGroq(id='req_01ketqp4e0fcc8bj9yv3w0s6jd', debug=None, seed=9638788, usage=None)) 

# Raw args: {"path":"./sample.txt"}

# Tool result (file content):
#  Life in the small coastal town moved at a rhythm that felt almost separate from the rest of the world. Each morning, the gulls announced the sunrise long before the first boats drifted out of the harbor, their calls echoing across the water like impatient reminders of the day ahead. Locals wandered the cobblestone streets with a familiarity born from generations living in the same place, exchanging quiet nods or warm greetings as they passed. And though nothing remarkable ever seemed to happen there, the town held a quiet magic—one woven from salty air, worn wooden docks, and stories whispered between waves.

#  FINAL SUMMARY: 
#  Here are 4 bullet points summarizing the content of './sample.txt':

# * The small coastal town has a unique rhythm that sets it apart from the rest of the world.
# * The town's daily life begins with the sound of gulls announcing the sunrise, followed by the departure of boats from the harbor.
# * The locals are deeply familiar with the town and its traditions, often greeting each other with nods or warm greetings as they pass by.
# * Despite the lack of remarkable events, the town has a quiet magic that is woven from its natural surroundings, including the salty air, wooden docks, and the stories shared among its residents.
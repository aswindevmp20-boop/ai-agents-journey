# Take messy instructions from the user
# Understand the purpose of the email
# Extract key details
# Write a clean, professional email
# Support multiple tones: formal, semi-formal, casual
# Use tool-calling to generate the final draft

import os,json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def write_email(subject: str, message_body:str, tone: str="formal"):
    if tone =="formal":
        greeting = "Dear Sir/Madam,"
        closing = "Regards,"
    elif tone == "semi-formal":
        greeting = "Hello,"
        closing = "Best Regards,"
    else:
        greeting = "Hi,"
        closing = "cheers,"

    email = f"""
{greeting}

{message_body}

{closing}
Your Name
"""

    return {"subject": subject, "email": email}

TOOL_FUNCTIONS = {
    "write_email" : write_email
}

tools = [
    {
        "type":"function",
        "function":{
            "name": "write_email",
            "description": "Generate a formatted email with subject, body and tone.",
            "parameters":{
                "type":"object",
                "properties":{
                    "subject":{"type":"string"},
                    "message_body":{"type":"string"},
                    "tone":{
                        "type":"string",
                        "enum":["formal", "semi-formal", "causal"],
                        "default":"formal"
                    }
                },
                "required":["subject","message_body","tone"]
            }
        }
    }
]

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You generate clean, structured email drafting instructions.\n"
        "Extract the subject, message body, and tone from the user's natural language.\n"
        "Tone defaults to 'formal' unless user says otherwise.\n"
        "Always call the write_email tool."
    )
}

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = messages,
        tools=tools
    )

def run_agent(query):
    messages = [SYSTEM_MSG, {"role":"user", "content":query}]

    while True:
        response = call_model(messages)
        msg = response.choices[0].message

        if msg.content:
            print("\n Final Answer: ", msg.content)
            break

        if msg.tool_calls:
            for call in msg.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments)

                print("\n Tool Called:", fn_name, "args:", args)

                result = TOOL_FUNCTIONS[fn_name](**args)
                print("End Result: ", result)

                messages.append({"role":"assistant", "tool_calls": msg.tool_calls})
                messages.append({
                    "role":"tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)
                })

query = """
My name is Shah rukh khan . write an email to my manager explaining that I need leave tomorrow because I have a medical appointment.
Tone should be semi-formal.
"""

run_agent(query)


# Expected Output:

# Tool Called: write_email args: {'message_body': "Hi, I'm Shah Rukh Khan and I need to take leave tomorrow because I have a medical appointment. I apologize for any inconvenience this may cause and will make sure to catch up on any missed work as soon as possible.", 'subject': 'Leave Request for Medical Appointment', 'tone': 'semi-formal'}
# End Result:  {'subject': 'Leave Request for Medical Appointment', 'email': "\nHello,\n\nHi, I'm Shah Rukh Khan and I need to take leave tomorrow because I have a medical appointment. I apologize for any inconvenience this may cause and will make sure to catch up on any missed work as soon as possible.\n\nBest Regards,\nYour Name\n"}

#  Final Answer:  Here is a formatted email with the subject, body, and tone you specified. 

# Subject: Leave Request for Medical Appointment

# Hello,

# Hi, I'm Shah Rukh Khan and I need to take leave tomorrow because I have a medical appointment. I apologize for any inconvenience this may cause and will make sure to catch up on any missed work as soon as possible.

# Best Regards,
# Shah Rukh Khan
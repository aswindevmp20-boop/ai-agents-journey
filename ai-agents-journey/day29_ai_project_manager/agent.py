from groq import Groq
import os, json
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def plan_project(goal):
    prompt = f"""
You are an AI Project manager.
Break this project into tasks.
Assign each task a priority: High, Medium, or Low.

Return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include text before or after JSON.

Format:
[
    {{"task":"...", "priority":"High","status":"Pending"}}
]

Project: {goal}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

def decide_task_agent(tasks):
    pending = [t for t in tasks if t["status"]=="Pending"]

    if not pending:
        return None
    
    priority_order = {"High":3, "Medium": 2, "Low": 1}
    pending.sort(key=lambda x: priority_order.get(x["priority"],0), reverse=True)
    return pending[0]

def run_agent(goal):
    print("\n Project:", goal)

    print("\n---Planning---")
    raw_plan = plan_project(goal)
    print(raw_plan)

    try:
        tasks = json.loads(raw_plan)

    except:
        print("\n Could not parse JSON. Try again.")
        return
    
    while True:
        next_task = decide_task_agent(tasks)
        
        if not next_task:
            print("\n Project completed")
            break

        print(f"\n Next Task (Priority: {next_task['priority']}): {next_task['task']}")
        input("Press enter to mark as completed...")

        next_task["status"] = "completed"

        print("\n--- CURRENT STATUS ---")
        for t in tasks:
            print(f"{t['task']} | {t['priority']} | {t['status']}")


while True:
    goal = input("\nEnter project goal (or exit): ")
    if goal.lower() == "exit":
        break
    run_agent(goal)


# Expected Output:-

# Enter project goal (or exit): To protect ocean.

#  Project: To protect ocean.

# ---Planning---
# [
#     {"task":"Conduct research on ocean pollution", "priority":"High","status":"Pending"},
#     {"task":"Identify areas of high pollution", "priority":"Medium","status":"Pending"},
#     {"task":"Develop a plan to reduce plastic waste", "priority":"High","status":"Pending"},
#     {"task":"Collaborate with organizations to implement plan", "priority":"High","status":"Pending"},
#     {"task":"Create educational materials to raise awareness", "priority":"Medium","status":"Pending"},
#     {"task":"Organize beach cleanups and community events", "priority":"Medium","status":"Pending"},
#     {"task":"Monitor and track progress of ocean conservation efforts", "priority":"Low","status":"Pending"},
#     {"task":"Establish partnerships with local businesses and governments", "priority":"High","status":"Pending"},
#     {"task":"Develop a social media campaign to raise awareness", "priority":"Low","status":"Pending"}
# ]

#  Next Task (Priority: High): Conduct research on ocean pollution
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | Pending
# Develop a plan to reduce plastic waste | High | Pending
# Collaborate with organizations to implement plan | High | Pending
# Create educational materials to raise awareness | Medium | Pending
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | Pending
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: High): Develop a plan to reduce plastic waste
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | Pending
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | Pending
# Create educational materials to raise awareness | Medium | Pending
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | Pending
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: High): Collaborate with organizations to implement plan
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | Pending
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | Pending
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | Pending
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: High): Establish partnerships with local businesses and governments
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | Pending
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | Pending
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: Medium): Identify areas of high pollution
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | completed
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | Pending
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: Medium): Create educational materials to raise awareness
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | completed
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | completed
# Organize beach cleanups and community events | Medium | Pending
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: Medium): Organize beach cleanups and community events
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | completed
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | completed
# Organize beach cleanups and community events | Medium | completed
# Monitor and track progress of ocean conservation efforts | Low | Pending
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: Low): Monitor and track progress of ocean conservation efforts
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | completed
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | completed
# Organize beach cleanups and community events | Medium | completed
# Monitor and track progress of ocean conservation efforts | Low | completed
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | Pending

#  Next Task (Priority: Low): Develop a social media campaign to raise awareness
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Conduct research on ocean pollution | High | completed
# Identify areas of high pollution | Medium | completed
# Develop a plan to reduce plastic waste | High | completed
# Collaborate with organizations to implement plan | High | completed
# Create educational materials to raise awareness | Medium | completed
# Organize beach cleanups and community events | Medium | completed
# Monitor and track progress of ocean conservation efforts | Low | completed
# Establish partnerships with local businesses and governments | High | completed
# Develop a social media campaign to raise awareness | Low | completed

#  Project completed

# Enter project goal (or exit):

#  Project:

# ---Planning---
# [
#     {"task":"Define project scope", "priority":"High","status":"Pending"},
#     {"task":"Identify project stakeholders", "priority":"Medium","status":"Pending"},
#     {"task":"Develop project timeline", "priority":"High","status":"Pending"},
#     {"task":"Create project budget", "priority":"Medium","status":"Pending"},
#     {"task":"Assign project team members", "priority":"High","status":"Pending"},
#     {"task":"Conduct project risk assessment", "priority":"Medium","status":"Pending"},
#     {"task":"Develop project communication plan", "priority":"Low","status":"Pending"},
#     {"task":"Create project dashboard", "priority":"Low","status":"Pending"},
#     {"task":"Establish project monitoring and control process", "priority":"Medium","status":"Pending"}
# ]

#  Next Task (Priority: High): Define project scope
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | Pending
# Develop project timeline | High | Pending
# Create project budget | Medium | Pending
# Assign project team members | High | Pending
# Conduct project risk assessment | Medium | Pending
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: High): Develop project timeline
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | Pending
# Develop project timeline | High | completed
# Create project budget | Medium | Pending
# Assign project team members | High | Pending
# Conduct project risk assessment | Medium | Pending
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: High): Assign project team members
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | Pending
# Develop project timeline | High | completed
# Create project budget | Medium | Pending
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | Pending
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: Medium): Identify project stakeholders
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | Pending
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | Pending
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: Medium): Create project budget
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | completed
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | Pending
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: Medium): Conduct project risk assessment
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | completed
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | completed
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | Pending

#  Next Task (Priority: Medium): Establish project monitoring and control process
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | completed
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | completed
# Develop project communication plan | Low | Pending
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | completed

#  Next Task (Priority: Low): Develop project communication plan
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | completed
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | completed
# Develop project communication plan | Low | completed
# Create project dashboard | Low | Pending
# Establish project monitoring and control process | Medium | completed

#  Next Task (Priority: Low): Create project dashboard
# Press enter to mark as completed...

# --- CURRENT STATUS ---
# Define project scope | High | completed
# Identify project stakeholders | Medium | completed
# Develop project timeline | High | completed
# Create project budget | Medium | completed
# Assign project team members | High | completed
# Conduct project risk assessment | Medium | completed
# Develop project communication plan | Low | completed
# Create project dashboard | Low | completed
# Establish project monitoring and control process | Medium | completed

#  Project completed
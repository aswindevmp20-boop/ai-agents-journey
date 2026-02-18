# We expose a tool:
# get_youtube_transcript(url)
# The agent will:
    # Detect that the user has given a YouTube URL
    # Call this tool
    # Python extracts transcript
    # LLM summarizes


import os,json,re
from groq import Groq
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        match = re.search(p,url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "Invalid Youtube URL."

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([item["text"] for item in transcript])
        return full_text
    except Exception as e:
        return f"Transcript not available: {str(e)}"

TOOL_FUNCTIONS = {
    "get_youtube_transcript" : get_youtube_transcript
}

tools = [
    {
        "type": "function",
        "function":{
            "name": "get_youtube_transcript",
            "description": "Fetch transcript from a Youtube URL.",
            "parameters": {
                "type":"object",
                "properties": {
                    "url":{"type":"string"}
                },
                "required":["url"]
            }
        }
    }
]

def call_model(messages):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = messages,
        tools = tools
    )

def run_agent(query):
    messages = [{"role":"user", "content": query}]

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

                print("\n Tool called:", fn_name, "args", args)
                result = TOOL_FUNCTIONS[fn_name](**args)

                messages.append({"role":"assistant", "tool_calls":msg.tool_calls})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result)
                    }
                )


query = "Summarize this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ"

run_agent(query)


# Expected Output:

# Tool called: get_youtube_transcript args {'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'}

#  Final Answer:  Unfortunately, the function was unable to retrieve the transcript for the given YouTube video. The video in question is the popular music video "Rickroll" by Rick Astley, and its transcript may not be publicly available or may be restricted due to copyright or other reasons.

# If you're looking for a summary of the video, I can provide a general overview. The video is a music video for the 1987 song "Never Gonna Give You Up" by Rick Astley. It features Astley singing and performing the song, with some dance moves and scenes of him walking around and interacting with people. The video has become a meme and cultural phenomenon, often used as a prank or joke to trick people into watching it. However, please note that the actual content of the video may not be suitable for all audiences.
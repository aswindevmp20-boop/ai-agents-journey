from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class QueryRequest(BaseModel):
    query: str

@app.post("/agent/")
def run_agent(request: QueryRequest):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are a helpful AI assistant."},
            {"role":"user","content":request.query}
        ]
    )

    return{
        "query":request.query,
        "result": response
    }




# Expected Output:-

# uvicorn main:app --reload

# Run via swagger:- 

# {
#   "query": "What is the use of decorators in fastapi",
#   "result": {
#     "id": "chatcmpl-e938be7a-5b0d-45af-b8a7-041a64e82c3c",
#     "choices": [
#       {
#         "finish_reason": "stop",
#         "index": 0,
#         "logprobs": null,
#         "message": {
#           "content": "**Introduction to Decorators in FastAPI**\n=====================================\n\nDecorators are a powerful tool in Python that can be used to modify or extend the behavior of a function or class. In the context of FastAPI, decorators are used to add additional functionality to routes, such as authentication, rate limiting, and error handling.\n\n**Common Use Cases for Decorators in FastAPI**\n-----------------------------------------\n\n1. **Authentication and Authorization**: Decorators can be used to check if a user is authenticated or has the required permissions to access a route.\n2. **Rate Limiting**: Decorators can be used to limit the number of requests that can be made to a route within a certain time frame.\n3. **Error Handling**: Decorators can be used to catch and handle errors that occur within a route.\n4. **Logging**: Decorators can be used to log information about requests and responses.\n5. **Caching**: Decorators can be used to cache responses to routes.\n\n**Example of a Decorator in FastAPI**\n----------------------------------\n\nHere is an example of a simple decorator that logs information about requests:\n```python\nfrom fastapi import FastAPI, Request\nfrom logging import Logger\n\napp = FastAPI()\n\ndef logging_decorator(func):\n    def wrapper(*args, **kwargs):\n        request: Request = args[0]\n        logger = Logger(\"fastapi\")\n        logger.info(f\"Request: {request.method} {request.url}\")\n        return func(*args, **kwargs)\n    return wrapper\n\n@app.get(\"/items/\")\n@logging_decorator\ndef read_items():\n    return [{\"name\": \"Item 1\"}, {\"name\": \"Item 2\"}]\n```\nIn this example, the `logging_decorator` function is used to log information about requests to the `/items/` route.\n\n**Decorators Provided by FastAPI**\n----------------------------------\n\nFastAPI provides several built-in decorators that can be used to add functionality to routes. Some examples include:\n\n1. `@app.get()`: Used to define a GET route.\n2. `@app.post()`: Used to define a POST route.\n3. `@app.put()`: Used to define a PUT route.\n4. `@app.delete()`: Used to define a DELETE route.\n5. `@app.exception_handler()`: Used to define a custom exception handler.\n\n**Custom Decorators**\n--------------------\n\nCustom decorators can be created to add specific functionality to routes. Here is an example of a custom decorator that checks if a user is authenticated:\n```python\nfrom fastapi import FastAPI, Request, HTTPException\nfrom fastapi.security import HTTPBasic, HTTPBasicCredentials\n\napp = FastAPI()\nsecurity = HTTPBasic()\n\ndef authenticated(func):\n    def wrapper(*args, **kwargs):\n        request: Request = args[0]\n        credentials: HTTPBasicCredentials = security.get_credentials(request)\n        if credentials:\n            if credentials.username == \"admin\" and credentials.password == \"password\":\n                return func(*args, **kwargs)\n        raise HTTPException(status_code=401, detail=\"Unauthorized\")\n    return wrapper\n\n@app.get(\"/items/\")\n@authenticated\ndef read_items():\n    return [{\"name\": \"Item 1\"}, {\"name\": \"Item 2\"}]\n```\nIn this example, the `authenticated` decorator is used to check if a user is authenticated before allowing access to the `/items/` route.",
#           "role": "assistant",
#           "annotations": null,
#           "executed_tools": null,
#           "function_call": null,
#           "reasoning": null,
#           "tool_calls": null
#         }
#       }
#     ],
#     "created": 1770707967,
#     "model": "llama-3.3-70b-versatile",
#     "object": "chat.completion",
#     "mcp_list_tools": null,
#     "service_tier": "on_demand",
#     "system_fingerprint": "fp_c06d5113ec",
#     "usage": {
#       "completion_tokens": 682,
#       "prompt_tokens": 51,
#       "total_tokens": 733,
#       "completion_time": 1.5387946110000001,
#       "completion_tokens_details": null,
#       "prompt_time": 0.002987309,
#       "prompt_tokens_details": null,
#       "queue_time": 0.048338861,
#       "total_time": 1.54178192
#     },
#     "usage_breakdown": null,
#     "x_groq": {
#       "id": "req_01kh36pp5qers8vjhgav03hrrn",
#       "debug": null,
#       "seed": 301803991,
#       "usage": null
#     }
#   }
# }
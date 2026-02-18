from rest_framework.decorators import api_view
from rest_framework.response import Response
from .agent import run_agent

@api_view(['POST'])
def agent_endpoint(request):
    query = request.data.get("query")

    if not query:
        return Response({"error":"Query is required"}, status=400)
    
    result = run_agent(query)

    return Response({
        "query":query,
        "response":result
    })


# http://127.0.0.1:8000/api/agent/
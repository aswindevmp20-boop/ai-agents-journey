from fastapi import APIRouter
from pydantic import BaseModel
from app.agents.agent import run_agent

router  = APIRouter()

class QueryRequest(BaseModel):
    query: str

@router.post("/agent/")
def agent_endpoint(request: QueryRequest):
    result = run_agent(request.query)
    return{
        "query": request.query,
        "response": result
    }
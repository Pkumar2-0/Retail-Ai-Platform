from fastapi import APIRouter
from pydantic import BaseModel

from agents.orchestrator import orchestrator

router = APIRouter()


class AgentChatRequest(BaseModel):
    query: str


@router.get("/agent-chat")

def agent_chat(query: str):

    result = orchestrator(query)

    return result


@router.post("/agent-chat")
def agent_chat_post(request: AgentChatRequest):
    return orchestrator(request.query)

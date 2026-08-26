import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth.users import current_active_user
from app.db.models import User
from app.graph.graph import get_graph

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


@router.post("")
async def chat(
    payload: ChatRequest,
    user: Annotated[User, Depends(current_active_user)],
):
    # TODO: rate limiting / concurrency: a single local LLM worker or queue will serialize concurrent user requests.
    user_id = str(user.id)
    conversation_id = payload.conversation_id or "default"
    thread_id = f"{user_id}:{conversation_id}"
    graph = await get_graph()

    state = {
        "messages": [{"role": "user", "content": payload.message}],
        "user_id": user_id,
        "thread_id": thread_id,
        "retry_count": 0,
    }

    async def event_stream():
        try:
            async for event in graph.astream_events(state, version="v1"):
                kind = event.get("event")
                data = event.get("data", {})
                if kind in {"on_chain_start", "on_chain_end", "on_chat_model_stream", "on_chat_model_end"}:
                    payload_event = {"event": kind, "data": data}
                    yield f"data: {json.dumps(payload_event)}\n\n"
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import ToolDefinition, User

router = APIRouter()


class ToolRegisterRequest(BaseModel):
    name: str
    description: str


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@router.get("")
async def list_tools(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    result = await session.execute(select(ToolDefinition))
    return [
        {"id": str(tool.id), "name": tool.name, "description": tool.description}
        for tool in result.scalars().all()
    ]


@router.post("", response_model=dict)
async def register_tool(
    payload: ToolRegisterRequest,
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    if not payload.name or not payload.description:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name and description are required")

    # TODO: model config -> generate embedding from the tool description, output list[float] of length VECTOR_DIMENSION.
    tool_embedding = [0.0 for _ in range(settings.vector_dimension)]

    tool = ToolDefinition(name=payload.name, description=payload.description, embedding=tool_embedding)
    session.add(tool)
    await session.commit()
    await session.refresh(tool)
    return {"id": str(tool.id), "name": tool.name, "description": tool.description}

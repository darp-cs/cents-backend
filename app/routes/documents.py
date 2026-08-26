import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pgvector.sqlalchemy import Vector
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.config import settings
from app.db.base import AsyncSessionLocal
from app.db.models import Document, User

router = APIRouter()


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/upload", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name required")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text found in document")

    # TODO: model config -> generate embedding for a chunk or combined document text, output list[float] with VECTOR_DIMENSION length.
    placeholder_embedding = [0.0 for _ in range(settings.vector_dimension)]
    stored = []

    for chunk in chunks:
        doc = Document(
            user_id=user.id,
            source_filename=file.filename,
            chunk_text=chunk[:4000],
            embedding=placeholder_embedding,
        )
        session.add(doc)
        stored.append({"id": str(doc.id), "source_filename": file.filename, "chunk_text": chunk[:4000]})

    await session.commit()
    return {"status": "ok", "count": len(stored), "documents": stored}


@router.get("", response_model=list[dict])
async def list_documents(
    user: Annotated[User, Depends(current_active_user)],
    session: AsyncSession = Depends(get_db_session),
):
    result = await session.execute(select(Document).where(Document.user_id == user.id))
    return [
        {
            "id": str(document.id),
            "source_filename": document.source_filename,
            "chunk_text": document.chunk_text,
            "created_at": document.created_at.isoformat(),
        }
        for document in result.scalars().all()
    ]

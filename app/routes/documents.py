import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.users import current_active_user
from app.db.base import AsyncSessionLocal
from app.db.models import Document, User
from app.vector_store import upsert_documents

router = APIRouter()


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


@router.post("/upload", response_model=dict)
async def upload_document(
    user: Annotated[User, Depends(current_active_user)],
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File name required")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No text found in document")

    stored = []
    chroma_items: list[tuple[str, str]] = []

    for chunk in chunks:
        doc_id = uuid.uuid4()
        chunk_text = chunk[:4000]
        doc = Document(
            id=doc_id,
            user_id=user.id,
            source_filename=file.filename,
            chunk_text=chunk_text,
        )
        session.add(doc)
        doc_id_str = str(doc_id)
        chroma_items.append((doc_id_str, chunk_text))
        stored.append({"id": doc_id_str, "source_filename": file.filename, "chunk_text": chunk_text})

    await session.commit()
    upsert_documents(str(user.id), file.filename, chroma_items)
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

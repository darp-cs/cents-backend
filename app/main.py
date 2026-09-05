from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth.users import auth_router, register_router, user_router
from app.config import settings
from app.db.base import init_db
from app.graph.graph import get_graph
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router
from app.routes.documents import router as documents_router
from app.routes.tools import router as tools_router
from app.vector_store import ensure_vector_store_ready

app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    await init_db()
    ensure_vector_store_ready()
    await get_graph()


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(register_router, prefix="/auth", tags=["auth"])
app.include_router(user_router, prefix="/users", tags=["users"])
app.include_router(conversations_router, prefix="/conversations", tags=["conversations"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(tools_router, prefix="/tools", tags=["tools"])

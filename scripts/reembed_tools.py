from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.base import AsyncSessionLocal, init_db
from app.db.models import ToolDefinition
from app.llm.client import LLMClientError, embed_texts
from app.vector_store import upsert_tool_embedding


def _tool_embedding_input(name: str, description: str) -> str:
    return f"{name}\n{description}".strip()


async def reembed_tools() -> tuple[int, int]:
    await init_db()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ToolDefinition).order_by(ToolDefinition.name.asc()))
        tools = list(result.scalars().all())

    if not tools:
        print("No tools found. Nothing to re-embed.")
        return 0, 0

    succeeded = 0
    failed = 0

    for tool in tools:
        try:
            vectors = await embed_texts(
                [_tool_embedding_input(tool.name, tool.description)],
                metadata={
                    "entity": "tool_definition",
                    "reason": "reembed_script",
                },
            )
        except LLMClientError as exc:
            failed += 1
            print(f"Failed to embed tool '{tool.name}' ({tool.id}): {exc}")
            continue

        if not vectors or not vectors[0]:
            failed += 1
            print(f"Failed to embed tool '{tool.name}' ({tool.id}): empty embedding response")
            continue

        upsert_tool_embedding(
            str(tool.id),
            name=tool.name,
            description=tool.description,
            embedding=vectors[0],
            enabled=bool(tool.enabled),
        )
        succeeded += 1

    return succeeded, failed


async def _main() -> int:
    succeeded, failed = await reembed_tools()
    print(f"Re-embedded tools: {succeeded} succeeded, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

from dataclasses import asdict, dataclass
from typing import Protocol

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import LLMRequestLog


@dataclass
class RequestLogEntry:
    endpoint: str
    status: str  # "success" | "error"
    error_type: str | None
    model: str | None
    input_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int


class RequestLogRepository(Protocol):
    async def add(self, entry: RequestLogEntry) -> None: ...


class SqlRequestLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entry: RequestLogEntry) -> None:
        self.session.add(LLMRequestLog(**asdict(entry)))
        await self.session.commit()


class InMemoryRequestLogRepository:
    """For tests and local runs without a database."""

    def __init__(self) -> None:
        self.entries: list[RequestLogEntry] = []

    async def add(self, entry: RequestLogEntry) -> None:
        self.entries.append(entry)


async def get_request_log_repository(
    session: AsyncSession = Depends(get_session),
) -> RequestLogRepository:
    return SqlRequestLogRepository(session)

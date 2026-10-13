from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class LLMRequestLog(Base):
    __tablename__ = "llm_request_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    endpoint: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20))  # "success" | "error"
    error_type: Mapped[str | None] = mapped_column(String(50))
    model: Mapped[str | None] = mapped_column(String(100))
    input_chars: Mapped[int]
    prompt_tokens: Mapped[int | None]
    completion_tokens: Mapped[int | None]
    latency_ms: Mapped[int]

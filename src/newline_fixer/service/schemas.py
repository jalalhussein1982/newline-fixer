"""Request and response bodies of design 6.1."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class FixRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _valid_unicode(cls, v: str) -> str:
        try:
            v.encode("utf-8")
        except UnicodeEncodeError as e:
            raise ValueError("text is not valid Unicode (lone surrogate)") from e
        return v


class FixStats(BaseModel):
    tokens: int
    gaps: int
    changed: int
    model: str
    latency_ms: float


class FixResponse(BaseModel):
    text: str
    stats: FixStats


class Health(BaseModel):
    status: str
    model: str
    ready: bool

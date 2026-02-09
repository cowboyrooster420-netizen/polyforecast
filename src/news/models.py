from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    source: str = ""
    url: str = ""
    published_at: datetime | None = None
    description: str = ""

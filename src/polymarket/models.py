from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Token(BaseModel):
    token_id: str
    outcome: str
    price: float = 0.0


class Market(BaseModel):
    condition_id: str
    question: str
    slug: str = ""
    description: str = ""
    end_date: datetime | None = None
    active: bool = True
    closed: bool = False
    resolved: bool = False
    resolution: str = ""
    volume: float = 0.0
    liquidity: float = 0.0
    tokens: list[Token] = Field(default_factory=list)
    category: str = ""
    image: str = ""

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else ""

    def outcome_price(self, outcome: str) -> float | None:
        for t in self.tokens:
            if t.outcome.lower() == outcome.lower():
                return t.price
        return None


class Event(BaseModel):
    event_id: str = ""
    title: str = ""
    slug: str = ""
    description: str = ""
    markets: list[Market] = Field(default_factory=list)
    category: str = ""

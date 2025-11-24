from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Job(BaseModel):
    """Represents a job listing from any source."""

    # Required fields
    url: str
    title: str
    company: str
    source: str

    # Optional fields
    salary_min: int | None = None
    salary_max: int | None = None
    date_posted: str | None = None
    location: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Set after vector search
    score: float | None = None
    is_fresh: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values for cleaner output."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def to_mongo_doc(self) -> dict[str, Any]:
        """Convert to MongoDB document format (includes all fields)."""
        return self.model_dump()

    @classmethod
    def from_mongo_doc(cls, doc: dict[str, Any]) -> Job:
        """Create Job from MongoDB document."""
        return cls(
            url=doc.get("url", ""),
            title=doc.get("title", ""),
            company=doc.get("company", ""),
            source=doc.get("source", ""),
            salary_min=doc.get("salary_min"),
            salary_max=doc.get("salary_max"),
            date_posted=doc.get("date_posted"),
            location=doc.get("location"),
            tags=doc.get("tags", []),
            score=doc.get("score"),
            is_fresh=doc.get("is_fresh"),
        )

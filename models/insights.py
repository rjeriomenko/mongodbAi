from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SkillGap(BaseModel):
    """Represents a skill gap between user and job requirements."""

    skill: str
    priority: Literal["essential", "supplemental"]
    description: str


class ActionStep(BaseModel):
    """Represents an actionable step in the career plan."""

    action: str
    timeline: str
    impact: Literal["high", "medium", "low"]
    url: str | None = None


class Insights(BaseModel):
    """Container for all career insights."""

    skill_gaps: list[SkillGap]
    action_plan: list[ActionStep]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "skill_gaps": [gap.model_dump() for gap in self.skill_gaps],
            "action_plan": [step.model_dump() for step in self.action_plan]
        }

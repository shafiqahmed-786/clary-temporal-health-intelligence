"""
schemas/user.py — UserProfile and onboarding data.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str = Field(..., min_length=1)
    name: str = ""
    age: int | None = Field(None, ge=0, le=130)
    location: str = ""
    onboarding_notes: str = Field(
        "", description="Free-text from onboarding — dietary habits, conditions, etc."
    )
    known_conditions: list[str] = Field(default_factory=list)
    known_allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
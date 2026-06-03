"""
Database table definitions for Echo2Equation.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID
from .database import Base


class Corpus(Base):
    """
    Stores natural-language math expressions and their LaTeX equivalents.
    Used as a searchable corpus for the 'Retry' feature.
    """
    __tablename__ = "corpus"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text = Column(Text, nullable=False)       # e.g., "integral of x squared"
    latex = Column(Text, nullable=False)      # e.g., "\\int x^2 dx"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Feedback(Base):
    """
    Stores user feedback for a given prediction.
    Tracks what the model produced, and how the user responded.
    """
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transcript_text = Column(Text, nullable=True)    # from Whisper
    generated_latex = Column(Text, nullable=True)    # from MathT5
    correct = Column(Boolean, nullable=True)         # True, False, or None (not clicked)
    retried = Column(Boolean, default=False, nullable=False)
    record_again = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

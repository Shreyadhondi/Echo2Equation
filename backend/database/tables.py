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
    Tracks what the model produced, what the user corrected,
    and how the user responded.
    """
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # User input / transcript
    transcript_text = Column(Text, nullable=True)

    # Model prediction
    generated_latex = Column(Text, nullable=True)

    # Human-provided correction, useful for future retraining
    corrected_latex = Column(Text, nullable=True)

    # True  = user accepted generated_latex
    # False = user said generated_latex was wrong
    # None  = no final correctness decision
    correct = Column(Boolean, nullable=True)

    retried = Column(Boolean, default=False, nullable=False)
    record_again = Column(Boolean, default=False, nullable=False)

    # Optional metadata
    audio_path = Column(Text, nullable=True)
    visual_path = Column(Text, nullable=True)
    corpus_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
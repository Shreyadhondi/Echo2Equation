"""
FastAPI backend for Echo2Equation.

Endpoints:
----------
1) POST /transcribe
   - Accepts uploaded audio.
   - Runs Whisper transcription.
   - Returns transcript text.

2) POST /to_latex
   - Accepts spoken-style math text.
   - Runs MathT5.
   - Returns raw and cleaned LaTeX.

3) GET /corpus/search
   - Searches the seeded corpus table.
   - Kept for optional future suggestion/retry features.

4) POST /feedback
   - Stores user feedback in PostgreSQL.
   - Supports correct/incorrect feedback.
   - Stores corrected LaTeX if user provides it.
   - Adds accepted/corrected examples into the corpus for future reuse.

Important design:
-----------------
The frontend and backend are loosely coupled.
The frontend talks to this backend only through REST API calls.
"""

from __future__ import annotations

import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Database
import psycopg2
from psycopg2.pool import SimpleConnectionPool

# HuggingFace / Transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Local LaTeX cleaner
from backend.latex_parser.latex_clean import clean_latex


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

API_TITLE = "Echo2Equation API"
API_VERSION = "0.1.0"

FRONTEND_ORIGINS = {
    "http://127.0.0.1:5500",
    "http://localhost:5500",
}

STORAGE_DIR = Path("storage")
AUDIO_DIR = STORAGE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "echo2eq_db"
DB_USER = "echo2eq"
DB_PASS = "echo2eq_pw"

# These values are overridden inside Docker using environment variables
# if you set them in docker-compose.yml.
import os

DB_HOST = os.getenv("DB_HOST", DB_HOST)
DB_PORT = int(os.getenv("DB_PORT", str(DB_PORT)))
DB_NAME = os.getenv("DB_NAME", DB_NAME)
DB_USER = os.getenv("DB_USER", DB_USER)
DB_PASS = os.getenv("DB_PASS", DB_PASS)

MATHT5_BASE = Path(os.getenv("MATHT5_DIR", "models/matht5_model"))
WHISPER_MODEL_DIR = Path(
    os.getenv(
        "WHISPER_MODEL_DIR",
        "models/whisper/models--Systran--faster-whisper-base.en",
    )
)

GEN_BEAMS = int(os.getenv("GEN_BEAMS", "5"))
GEN_MAXLEN = int(os.getenv("GEN_MAXLEN", "128"))


# -----------------------------------------------------------------------------
# Whisper backend detection
# -----------------------------------------------------------------------------

HAVE_FASTER_WHISPER = False
HAVE_OPENAI_WHISPER = False

try:
    from faster_whisper import WhisperModel  # type: ignore

    HAVE_FASTER_WHISPER = True
except Exception:
    pass

if not HAVE_FASTER_WHISPER:
    try:
        import whisper  # type: ignore

        HAVE_OPENAI_WHISPER = True
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Utility: resolve trained MathT5 directory
# -----------------------------------------------------------------------------

def resolve_matht5_dir(base: Path) -> Path:
    """
    Resolve the actual trained MathT5 model folder.

    Supports:
    1. Flat layout:
       models/matht5_model/config.json
       models/matht5_model/model.safetensors

    2. Older nested layout:
       models/matht5_model/<run-folder>/
    """
    if (base / "config.json").exists() and (base / "model.safetensors").exists():
        return base

    subdirs = [p for p in base.glob("*") if p.is_dir()]

    if not subdirs:
        raise FileNotFoundError(f"No MathT5 model found in {base}")

    return max(subdirs, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------

class TranscribeResponse(BaseModel):
    transcript: str
    audio_path: Optional[str] = None
    duration_ms: Optional[int] = None


class ToLatexRequest(BaseModel):
    text: str = Field(..., description="Transcript text to convert into LaTeX")


class ToLatexResponse(BaseModel):
    raw_latex: str
    cleaned_latex: str
    latency_ms: int
    model_version: str


class CorpusHit(BaseModel):
    corpus_id: str
    text: str
    latex: str
    score: float


class FeedbackRequest(BaseModel):
    """
    Feedback sent by the frontend.

    correct=True:
        User accepted generated_latex.

    correct=False + corrected_latex:
        User rejected generated_latex and provided a correction.

    correct=False + no corrected_latex:
        User rejected output but skipped correction.
    """

    transcript_text: Optional[str] = None
    generated_latex: Optional[str] = None
    corrected_latex: Optional[str] = None
    correct: Optional[bool] = None

    # Kept for compatibility/future features.
    retried: bool = False
    record_again: bool = False

    audio_path: Optional[str] = None
    visual_path: Optional[str] = None
    corpus_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str


# -----------------------------------------------------------------------------
# App lifespan
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models and create database connection pool once at startup.
    """
    hf_logging.set_verbosity_error()

    # Database pool
    app.state.db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=6,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

    # Ensure runtime DB schema is compatible.
    ensure_runtime_schema()

    # MathT5
    matht5_dir = resolve_matht5_dir(MATHT5_BASE)
    app.state.matht5_dir = matht5_dir

    # use_fast=False avoids tokenizer loading issues for T5/SentencePiece.
    app.state.tok = AutoTokenizer.from_pretrained(
        str(matht5_dir),
        use_fast=False,
    )

    app.state.matht5 = AutoModelForSeq2SeqLM.from_pretrained(str(matht5_dir))
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.matht5.to(app.state.device).eval()

    # Whisper
    app.state.whisper_kind = None

    if HAVE_FASTER_WHISPER:
        model_path = str(WHISPER_MODEL_DIR) if WHISPER_MODEL_DIR.exists() else "base.en"

        app.state.whisper = WhisperModel(
            model_path,
            device=app.state.device,
            compute_type="int8_float16" if app.state.device == "cuda" else "int8",
        )
        app.state.whisper_kind = "faster-whisper"

    elif HAVE_OPENAI_WHISPER:
        app.state.whisper = whisper.load_model("base.en", device=app.state.device)
        app.state.whisper_kind = "openai-whisper"

    else:
        app.state.whisper = None
        app.state.whisper_kind = "none"

    print(
        f"[startup] DB=postgres://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME} | "
        f"MathT5={matht5_dir} | Whisper={app.state.whisper_kind} | "
        f"Device={app.state.device}"
    )

    try:
        yield
    finally:
        pool = app.state.db_pool
        if pool:
            pool.closeall()


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(FRONTEND_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Database helpers
# -----------------------------------------------------------------------------

def db_conn():
    """
    Get a database connection from the pool.
    """
    return app.state.db_pool.getconn()


def db_put(conn) -> None:
    """
    Return a database connection to the pool.
    """
    app.state.db_pool.putconn(conn)


def ensure_runtime_schema() -> None:
    """
    Make sure feedback/corpus tables have the columns required by the app.

    This protects the app when an old database volume already exists.
    It does not delete any existing data.
    """
    conn = app.state.db_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus (
                  id UUID PRIMARY KEY,
                  text TEXT NOT NULL,
                  latex TEXT NOT NULL,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  UNIQUE (text, latex)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                  id UUID PRIMARY KEY,
                  transcript_text TEXT,
                  generated_latex TEXT,
                  corrected_latex TEXT,
                  correct BOOLEAN,
                  retried BOOLEAN NOT NULL DEFAULT FALSE,
                  record_again BOOLEAN NOT NULL DEFAULT FALSE,
                  audio_path TEXT,
                  visual_path TEXT,
                  corpus_id UUID,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cur.execute(
                """
                ALTER TABLE feedback
                ADD COLUMN IF NOT EXISTS corrected_latex TEXT,
                ADD COLUMN IF NOT EXISTS audio_path TEXT,
                ADD COLUMN IF NOT EXISTS visual_path TEXT,
                ADD COLUMN IF NOT EXISTS corpus_id UUID;
                """
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        app.state.db_pool.putconn(conn)


def insert_into_corpus(cur, text: Optional[str], latex: Optional[str]) -> None:
    """
    Insert a verified spoken-text-to-LaTeX pair into the corpus.

    Used when:
    1. User clicks Correct.
    2. User clicks Incorrect and provides corrected LaTeX.

    ON CONFLICT prevents duplicate pairs.
    """
    clean_text = (text or "").strip()
    clean_output = (latex or "").strip()

    if not clean_text or not clean_output:
        return

    cur.execute(
        """
        INSERT INTO corpus (id, text, latex)
        VALUES (%s, %s, %s)
        ON CONFLICT (text, latex) DO NOTHING;
        """,
        (str(uuid.uuid4()), clean_text, clean_output),
    )


# -----------------------------------------------------------------------------
# Corpus search helpers
# -----------------------------------------------------------------------------

STOP_WORDS = {
    "log",
    "ln",
    "sin",
    "cos",
    "tan",
    "csc",
    "sec",
    "cot",
    "sqrt",
    "frac",
    "sum",
    "int",
    "pi",
    "exp",
    "times",
    "over",
    "plus",
    "minus",
    "equals",
    "equal",
    "of",
    "the",
    "and",
}


def normalize_query(q: str) -> List[str]:
    """
    Convert a search query into simple searchable tokens.
    """
    tokens = re.findall(r"[A-Za-z]+", q.lower())
    filtered = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]

    seen = set()
    output = []

    for token in filtered:
        if token not in seen:
            seen.add(token)
            output.append(token)

    return output


# -----------------------------------------------------------------------------
# Endpoint: /transcribe
# -----------------------------------------------------------------------------

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Recorded audio blob"),
    ext: Optional[str] = Form(None),
):
    """
    Accept uploaded audio and return Whisper transcript.
    """
    if app.state.whisper is None:
        raise HTTPException(
            status_code=503,
            detail="Whisper is not available on the server.",
        )

    suggested_ext = (
        ext or Path(audio.filename or "").suffix.lstrip(".") or "webm"
    ).lower()

    if suggested_ext not in {"webm", "wav", "mp3", "m4a"}:
        suggested_ext = "webm"

    uid = uuid.uuid4().hex[:8]
    out_path = AUDIO_DIR / f"rec_{int(time.time())}_{uid}.{suggested_ext}"

    with out_path.open("wb") as f:
        f.write(await audio.read())

    transcript = ""
    duration_ms: Optional[int] = None

    try:
        if app.state.whisper_kind == "faster-whisper":
            segments, info = app.state.whisper.transcribe(
                str(out_path),
                beam_size=5,
                vad_filter=True,
            )

            transcript = " ".join(seg.text for seg in segments).strip()

            if info and getattr(info, "duration", None) is not None:
                duration_ms = int(info.duration * 1000)

        elif app.state.whisper_kind == "openai-whisper":
            result = app.state.whisper.transcribe(str(out_path))
            transcript = (result.get("text") or "").strip()

        else:
            raise RuntimeError("No Whisper backend configured.")

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Whisper transcription error: {exc!s}",
        )

    return TranscribeResponse(
        transcript=transcript,
        audio_path=str(out_path),
        duration_ms=duration_ms,
    )


# -----------------------------------------------------------------------------
# Endpoint: /to_latex
# -----------------------------------------------------------------------------

@app.post("/to_latex", response_model=ToLatexResponse)
async def to_latex(req: ToLatexRequest):
    """
    Convert spoken-style math text to LaTeX using MathT5.
    """
    text = (req.text or "").strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Field 'text' is required and cannot be empty.",
        )

    tokenizer = app.state.tok
    model = app.state.matht5
    device = app.state.device

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_length=GEN_MAXLEN,
            num_beams=GEN_BEAMS,
            early_stopping=True,
        )

    latency_ms = int((time.perf_counter() - start_time) * 1000.0)

    raw_latex = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    cleaned_latex = clean_latex(raw_latex, keep_dollars=True)

    return ToLatexResponse(
        raw_latex=raw_latex,
        cleaned_latex=cleaned_latex,
        latency_ms=latency_ms,
        model_version=str(app.state.matht5_dir),
    )


# -----------------------------------------------------------------------------
# Endpoint: /corpus/search
# -----------------------------------------------------------------------------

@app.get("/corpus/search", response_model=List[CorpusHit])
async def corpus_search(
    q: str = Query(..., min_length=1, description="Query string from transcript"),
    limit: int = Query(5, ge=1, le=25),
):
    """
    Search the corpus table.

    This endpoint is currently optional.
    It can be used later for top-k suggestions.
    """
    tokens = normalize_query(q)

    if not tokens:
        return []

    where = " AND ".join(["LOWER(text) LIKE %s"] * len(tokens))
    params = [f"%{token}%" for token in tokens]

    sql = f"""
        SELECT id, text, latex
        FROM corpus
        WHERE {where}
        LIMIT %s;
    """

    conn = db_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (*params, limit))
            rows = cur.fetchall()

    finally:
        db_put(conn)

    return [
        CorpusHit(
            corpus_id=str(row[0]),
            text=row[1],
            latex=row[2],
            score=1.0,
        )
        for row in rows
    ]


# -----------------------------------------------------------------------------
# Endpoint: /feedback
# -----------------------------------------------------------------------------

@app.post("/feedback", response_model=FeedbackResponse)
async def save_feedback(req: FeedbackRequest):
    """
    Store user feedback and optionally add verified examples to the corpus.

    Cases:
    ------
    1. User clicks Correct:
       correct=True
       generated_latex is inserted into corpus.

    2. User clicks Incorrect and provides correction:
       correct=False
       corrected_latex is inserted into corpus.

    3. User clicks Incorrect and skips correction:
       correct=False
       feedback is stored, but corpus is not updated.
    """
    feedback_id = str(uuid.uuid4())

    transcript = (req.transcript_text or "").strip() or None
    generated_latex = (req.generated_latex or "").strip() or None
    corrected_latex = (req.corrected_latex or "").strip() or None

    # Clean corrected LaTeX before saving if user entered it.
    if corrected_latex:
        corrected_latex = clean_latex(corrected_latex, keep_dollars=True)

    insert_feedback_sql = """
        INSERT INTO feedback
          (
            id,
            transcript_text,
            generated_latex,
            corrected_latex,
            correct,
            retried,
            record_again,
            audio_path,
            visual_path,
            corpus_id
          )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
    """

    conn = db_conn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                insert_feedback_sql,
                (
                    feedback_id,
                    transcript,
                    generated_latex,
                    corrected_latex,
                    req.correct,
                    req.retried,
                    req.record_again,
                    req.audio_path,
                    req.visual_path,
                    req.corpus_id,
                ),
            )

            # If user accepted the model output, add that pair to corpus.
            if req.correct is True:
                insert_into_corpus(cur, transcript, generated_latex)

            # If user rejected model output but supplied correction,
            # add corrected pair to corpus.
            elif req.correct is False and corrected_latex:
                insert_into_corpus(cur, transcript, corrected_latex)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        db_put(conn)

    return FeedbackResponse(id=feedback_id)
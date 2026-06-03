"""
FastAPI backend

Exposes four endpoints for the frontend:
  1) POST /transcribe        -> run Whisper on uploaded audio (multipart/form-data)
  2) POST /to_latex          -> run MathT5 on transcript text (JSON)
  3) GET  /corpus/search     -> search seeded corpus in Postgres (retry helper)
  4) POST /feedback          -> record user's feedback flags in Postgres (JSON)

Design notes:
- Models (Whisper + MathT5) are loaded once at startup and reused for all requests.
- DB connections use a small psycopg2 pool.
- All responses are JSON; LaTeX returned from /to_latex is already cleaned.
- Keep endpoints small and deterministic; the frontend owns the UI/UX state machine.

You can safely modify any default paths/ports below via environment variables.
"""

from __future__ import annotations

import os
import io
import re
import time
import uuid
import json
import shutil
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# DB
import psycopg2
from psycopg2.pool import SimpleConnectionPool

# HF / Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.utils import logging as hf_logging

# Our LaTeX cleaner (already in your repo)
from backend.latex_parser.latex_clean import clean_latex


# -----------------------------------------------------------------------------
# Configuration (env with sensible defaults for local dev)
# -----------------------------------------------------------------------------
API_TITLE = "Echo2Equation API"
API_VERSION = "0.1.0"

# Frontend origins for CORS (adjust if you use a different port)
FRONTEND_ORIGINS = {
    os.getenv("FRONTEND_ORIGIN_1", "http://127.0.0.1:5500"),
    os.getenv("FRONTEND_ORIGIN_2", "http://localhost:5500"),
}

# Storage locations
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "storage"))
AUDIO_DIR = STORAGE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Database (matches your docker-compose/.env; you can export these locally)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "echo2eq_db")
DB_USER = os.getenv("DB_USER", "echo2eq")
DB_PASS = os.getenv("DB_PASS", "echo2eq_pw")

# MathT5 model location (your trained weights)
MATHT5_BASE = Path(os.getenv("MATHT5_DIR", "models/matht5_model"))

# Whisper model location
# If you have the faster-whisper CTranslate2 model locally, point WHISPER_MODEL_DIR to that folder.
# Otherwise, we fall back to the model name "base.en" (downloaded automatically).
WHISPER_MODEL_DIR = Path(os.getenv("WHISPER_MODEL_DIR", "models/whisper/models--Systran--faster-whisper-base.en"))

# Beam size & lengths for generation
GEN_BEAMS = int(os.getenv("GEN_BEAMS", "5"))
GEN_MAXLEN = int(os.getenv("GEN_MAXLEN", "128"))


# -----------------------------------------------------------------------------
# Optional: faster-whisper vs openai-whisper detection
# -----------------------------------------------------------------------------
HAVE_FASTER_WHISPER = False
HAVE_OPENAI_WHISPER = False

try:
    # pip install faster-whisper
    from faster_whisper import WhisperModel  # type: ignore
    HAVE_FASTER_WHISPER = True
except Exception:
    pass

if not HAVE_FASTER_WHISPER:
    try:
        # pip install openai-whisper  (aka "whisper")
        import whisper  # type: ignore
        HAVE_OPENAI_WHISPER = True
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Utility: resolve MathT5 directory (handles both "flat" and "run-subdir" layouts)
# -----------------------------------------------------------------------------
def resolve_matht5_dir(base: Path) -> Path:
    # Case 1: files directly under models/matht5_model
    if (base / "config.json").exists() and (base / "model.safetensors").exists():
        return base
    # Case 2: older layout: choose most-recent subdir
    subs = [p for p in base.glob("*") if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No MathT5 model found in {base}")
    return max(subs, key=lambda p: p.stat().st_mtime)


# -----------------------------------------------------------------------------
# Pydantic Schemas
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
    transcript_text: Optional[str] = None
    generated_latex: Optional[str] = None
    correct: Optional[bool] = None
    retried: bool = False
    record_again: bool = False
    audio_path: Optional[str] = None
    visual_path: Optional[str] = None
    corpus_id: Optional[str] = None  # UUID string referencing corpus(id)


class FeedbackResponse(BaseModel):
    id: str


# -----------------------------------------------------------------------------
# App lifespan: load models + create DB pool once, and reuse them
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    hf_logging.set_verbosity_error()  # hush Transformers logs

    # --- DB pool ---
    app.state.db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=6,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

    # --- MathT5 ---
    matht5_dir = resolve_matht5_dir(MATHT5_BASE)
    app.state.matht5_dir = matht5_dir
    app.state.tok = AutoTokenizer.from_pretrained(
        str(matht5_dir),
        use_fast=False,
    )
    app.state.matht5 = AutoModelForSeq2SeqLM.from_pretrained(str(matht5_dir))
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.matht5.to(app.state.device).eval()

    # --- Whisper (prefer faster-whisper if available) ---
    app.state.whisper_kind = None
    if HAVE_FASTER_WHISPER:
        # You can tailor compute_type depending on your hardware
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
        f"MathT5={matht5_dir} | Whisper={app.state.whisper_kind} | Device={app.state.device}"
    )

    try:
        yield
    finally:
        # Close DB pool on shutdown
        pool = app.state.db_pool
        if pool:
            pool.closeall()


app = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)

# CORS: allow your dev frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(FRONTEND_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def db_conn():
    """Get a connection from the pool (remember to close it)."""
    return app.state.db_pool.getconn()


def db_put(conn):
    """Return a connection to the pool."""
    app.state.db_pool.putconn(conn)


# Very light "search token" sanitizer: drop digits, single letters, and common math words
STOP_WORDS = {
    "log", "ln", "sin", "cos", "tan", "csc", "sec", "cot",
    "sqrt", "frac", "sum", "int", "pi", "exp", "times", "over",
    "plus", "minus", "equals", "equal", "of", "the", "and",
}

def normalize_query(q: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z]+", q.lower())
    filt = [t for t in tokens if len(t) > 1 and t not in STOP_WORDS]
    # Deduplicate but keep order
    seen = set()
    out = []
    for t in filt:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# -----------------------------------------------------------------------------
# Endpoint: /transcribe  (POST, multipart/form-data)
# -----------------------------------------------------------------------------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    audio: UploadFile = File(..., description="Recorded audio blob"),
    ext: Optional[str] = Form(None),
):
    """
    Accepts an uploaded audio file and returns Whisper transcript.

    - Saves the file under storage/audio/<timestamp>_<uuid>.<ext>
    - Supports webm/opus and wav (others may work if ffmpeg is available)
    - Uses faster-whisper if installed, otherwise openai-whisper; if neither is
      available, returns HTTP 503 with an explanatory message.
    """
    if app.state.whisper is None:
        raise HTTPException(status_code=503, detail="Whisper is not available on the server.")

    # Infer extension from filename or form field
    suggested_ext = (ext or Path(audio.filename or "").suffix.lstrip(".") or "webm").lower()
    if suggested_ext not in {"webm", "wav", "mp3", "m4a"}:
        suggested_ext = "webm"

    # Save to disk
    uid = uuid.uuid4().hex[:8]
    out_path = AUDIO_DIR / f"rec_{int(time.time())}_{uid}.{suggested_ext}"
    with out_path.open("wb") as f:
        # UploadFile is async, but file chunks are small; reading all at once is fine here
        f.write(await audio.read())

    # Transcribe
    t0 = time.perf_counter()
    transcript = ""
    duration_ms: Optional[int] = None

    try:
        if app.state.whisper_kind == "faster-whisper":
            # faster-whisper returns an iterator over segments + info
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
            # openai-whisper doesn't provide duration here; leave None
        else:
            raise RuntimeError("No Whisper backend configured.")
    except Exception as e:
        # If decoding fails (e.g., ffmpeg missing), surface a friendly error
        raise HTTPException(status_code=500, detail=f"Whisper transcription error: {e!s}")

    # Done
    _ = (time.perf_counter() - t0) * 1000.0
    return TranscribeResponse(transcript=transcript, audio_path=str(out_path), duration_ms=duration_ms)


# -----------------------------------------------------------------------------
# Endpoint: /to_latex  (POST, JSON)
# -----------------------------------------------------------------------------
@app.post("/to_latex", response_model=ToLatexResponse)
async def to_latex(req: ToLatexRequest):
    """
    Runs MathT5 to convert a spoken-style math sentence to LaTeX.

    Returns both the raw model output and a cleaned version (wrapped in $...$),
    along with a small metadata block.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail="Field 'text' is required and cannot be empty.")

    tok = app.state.tok
    model = app.state.matht5
    device = app.state.device

    # Generate
    enc = tok(text, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=GEN_MAXLEN,
            num_beams=GEN_BEAMS,
            early_stopping=True,
        )
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    raw = tok.decode(out[0], skip_special_tokens=True).strip()
    cleaned = clean_latex(raw, keep_dollars=True)

    # model_version is helpful for debugging (e.g., folder name)
    model_version = str(app.state.matht5_dir)

    return ToLatexResponse(
        raw_latex=raw,
        cleaned_latex=cleaned,
        latency_ms=latency_ms,
        model_version=model_version,
    )


# -----------------------------------------------------------------------------
# Endpoint: /corpus/search  (GET, query string)
# -----------------------------------------------------------------------------
@app.get("/corpus/search", response_model=List[CorpusHit])
async def corpus_search(
    q: str = Query(..., min_length=1, description="Query string from transcript"),
    limit: int = Query(5, ge=1, le=25),
):
    """
    Searches the 'corpus' table for similar entries.

    Server sanitizes `q`:
      - removes digits
      - removes single-letter variables
      - removes common math words (STOP_WORDS)
    Then performs a simple ILIKE AND-query across the remaining tokens.

    If pg_trgm/tsvector is available later, you can upgrade this to a ranked query.
    """
    tokens = normalize_query(q)
    if not tokens:
        return []

    where = " AND ".join(["LOWER(text) LIKE %s"] * len(tokens))
    params = [f"%{t}%" for t in tokens]

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

    # naive "score": fraction of tokens present (since we used AND, it's always 1.0)
    hits = [
        CorpusHit(
            corpus_id=str(r[0]),
            text=r[1],
            latex=r[2],
            score=1.0,
        )
        for r in rows
    ]
    return hits


# -----------------------------------------------------------------------------
# Endpoint: /feedback  (POST, JSON)
# -----------------------------------------------------------------------------
@app.post("/feedback", response_model=FeedbackResponse)
async def save_feedback(req: FeedbackRequest):
    """
    Inserts a feedback row. Only a few fields are strictly required; others are optional.
    The DB schema was created earlier via your migration/seed scripts.

    Columns we write:
      id (UUID PK),
      transcript_text TEXT,
      generated_latex TEXT,
      correct BOOLEAN,
      retried BOOLEAN,
      record_again BOOLEAN,
      audio_path TEXT,
      visual_path TEXT,
      corpus_id UUID (nullable, FK to corpus.id),
      created_at TIMESTAMPTZ DEFAULT NOW()
    """
    fid = str(uuid.uuid4())

    sql = """
        INSERT INTO feedback
          (id, transcript_text, generated_latex, correct, retried, record_again,
           audio_path, visual_path, corpus_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """

    conn = db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    fid,
                    req.transcript_text,
                    req.generated_latex,
                    req.correct,
                    req.retried,
                    req.record_again,
                    req.audio_path,
                    req.visual_path,
                    req.corpus_id,
                ),
            )
        conn.commit()
    finally:
        db_put(conn)

    return FeedbackResponse(id=fid)

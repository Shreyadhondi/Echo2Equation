import os, json, uuid, psycopg2
from pathlib import Path

# <-- change if your file name/path is different
DATA_FILE = Path("data/full_dataset.jsonl")

DB_USER = os.getenv("DB_USER", "echo2eq")
DB_PASS = os.getenv("DB_PASS", "echo2eq_pw")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "echo2eq_db")

DDL_1 = """
CREATE TABLE IF NOT EXISTS corpus (
  id UUID PRIMARY KEY,
  text TEXT NOT NULL,
  latex TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (text, latex)
);
"""
DDL_2 = """
CREATE TABLE IF NOT EXISTS feedback (
  id UUID PRIMARY KEY,
  transcript_text TEXT,
  generated_latex TEXT,
  correct BOOLEAN,
  retried BOOLEAN NOT NULL DEFAULT FALSE,
  record_again BOOLEAN NOT NULL DEFAULT FALSE,
  audio_path TEXT,
  visual_path TEXT,
  corpus_id UUID,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
DDL_3 = """
ALTER TABLE feedback
ADD COLUMN IF NOT EXISTS audio_path TEXT,
ADD COLUMN IF NOT EXISTS visual_path TEXT,
ADD COLUMN IF NOT EXISTS corpus_id UUID;
"""

def main():
    if not DATA_FILE.exists():
        print(f"❌ Seed file not found: {DATA_FILE.resolve()}")
        return

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        host=DB_HOST, port=DB_PORT
    )
    conn.autocommit = False
    cur = conn.cursor()

    # ensure tables exist
    cur.execute(DDL_1); cur.execute(DDL_2); cur.execute(DDL_3); conn.commit()

    insert_sql = """
        INSERT INTO corpus (id, text, latex)
        VALUES (%s, %s, %s)
        ON CONFLICT (text, latex) DO NOTHING;
    """

    processed = 0
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text  = (obj.get("spoken") or "").strip()
                latex = (obj.get("latex")  or "").strip()
                if not text or not latex:
                    continue
                cur.execute(insert_sql, (str(uuid.uuid4()), text, latex))
                processed += 1
                if processed % 1000 == 0:
                    conn.commit()
                    print(f"[progress] inserted: {processed:,}")
            except Exception as e:
                print(f"[skip line {i}] {e}")

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM corpus;")
    total = cur.fetchone()[0]
    cur.close(); conn.close()

    print(f"✅ Done. Corpus total rows in DB: {total:,}")

if __name__ == "__main__":
    main()

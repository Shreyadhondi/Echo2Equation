import os, uuid, argparse, psycopg2

DB_USER = os.getenv("DB_USER", "echo2eq")
DB_PASS = os.getenv("DB_PASS", "echo2eq_pw")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "echo2eq_db")

def str2bool(v):
    if v is None: return None
    s = str(v).lower()
    if s in ("1","true","t","yes","y"): return True
    if s in ("0","false","f","no","n"): return False
    raise argparse.ArgumentTypeError("expected true/false")

def main():
    ap = argparse.ArgumentParser(description="Insert a feedback row.")
    ap.add_argument("--transcript", default=None)
    ap.add_argument("--generated_latex", default=None)
    ap.add_argument("--correct", type=str2bool, default=None)      # true/false
    ap.add_argument("--retried", type=str2bool, default=False)     # true/false
    ap.add_argument("--record_again", type=str2bool, default=False)
    ap.add_argument("--audio", dest="audio_path", default=None)
    ap.add_argument("--visual", dest="visual_path", default=None)
    ap.add_argument("--corpus_id", default=None)  # UUID of a corpus row
    args = ap.parse_args()

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS,
                            host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    sql = """
    INSERT INTO feedback
      (id, transcript_text, generated_latex, correct, retried, record_again,
       audio_path, visual_path, corpus_id)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    RETURNING id;
    """
    fid = str(uuid.uuid4())
    cur.execute(sql, (fid, args.transcript, args.generated_latex, args.correct,
                      args.retried, args.record_again, args.audio_path,
                      args.visual_path, args.corpus_id))
    conn.commit()
    print("Inserted feedback id:", fid)
    cur.close(); conn.close()

if __name__ == "__main__":
    main()

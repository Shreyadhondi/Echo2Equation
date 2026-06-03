import os, sys, psycopg2

DB_USER = os.getenv("DB_USER", "echo2eq")
DB_PASS = os.getenv("DB_PASS", "echo2eq_pw")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "echo2eq_db")

def main():
    if len(sys.argv) < 2:
        print("Usage: python database/search_corpus.py \"your query\" [limit]")
        return
    q = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS,
                            host=DB_HOST, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""SELECT id, text, latex
                   FROM corpus
                   WHERE text ILIKE %s
                   LIMIT %s;""", (f"%{q}%", limit))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No matches.")
        return
    for i, (cid, text, latex) in enumerate(rows, 1):
        t = (text[:90] + "…") if len(text) > 90 else text
        l = (latex[:90] + "…") if len(latex) > 90 else latex
        print(f"{i}. id={cid}  text={t}  latex={l}")

if __name__ == "__main__":
    main()


#--------------------
# TO RUN: Run from the root 
#---------------------
#export DB_HOST=localhost DB_USER=echo2eq DB_PASS=echo2eq_pw DB_NAME=echo2eq_db DB_PORT=5432
#python database/search_corpus.py "integral" 5
#-------------------------
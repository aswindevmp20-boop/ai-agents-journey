import hashlib
from shared.db import get_connection

def hash_query(query: str):
    return hashlib.sha256(query.encode()).hexdigest()

def check_cache(query:str):
    query_hash = hash_query(query)
    
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT result FROM jobs WHERE id=%s AND status='completed'",
        (query_hash,)
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return row[0]
    
    return None
import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host="postgres",
        database="ai_jobs",
        user="ai_user",
        password="ai_pass"
    )

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    #Existing jobs table...

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
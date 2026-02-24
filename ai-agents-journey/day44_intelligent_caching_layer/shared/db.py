import psycopg2
import os

def get_connection():
    return psycopg2.connect(
        host="postgres",
        database="ai_jobs",
        user="ai_user",
        password="ai_pass"
    )
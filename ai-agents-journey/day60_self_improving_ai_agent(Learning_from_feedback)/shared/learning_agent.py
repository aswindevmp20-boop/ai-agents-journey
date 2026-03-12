from shared.db import get_connection

def analyze_feedback():

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
SELECT rating, comment
FROM feedback
ORDER BY created_at DESC
LIMIT 50"""
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    positive = [r for r in rows if r[0] >= 4]
    negative = [r for r in rows if r[0] <= 2]

    return {
        "positive_feedback": len(positive),
        "negative_feedback": len(negative),
        "recent_feedback": rows
    }

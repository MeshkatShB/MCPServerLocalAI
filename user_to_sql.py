# fastmcp_sql_server.py

import sqlite3
import json
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage

DB_PATH = "mydb.sqlite"

def introspect_schema(db_path: str) -> dict:
    """Read all tables and their columns into a dict."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # 1. Get all table names
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        schema = {}
        # 2. For each table, fetch column details
        for table in tables:
            cur.execute(f"PRAGMA table_info('{table}')")
            cols = [
                {
                    "cid":   col[0],
                    "name":  col[1],
                    "type":  col[2],
                    "notnull": bool(col[3]),
                    "dflt_value": col[4],
                    "pk":    bool(col[5])
                }
                for col in cur.fetchall()
            ]
            schema[table] = cols
    finally:
        conn.close()
    return schema

# 1️⃣ Instantiate the MCP server
mcp = FastMCP(name="SQLAgent")

# 2️⃣ Prompt: inject the live schema into the LLM’s instruction
@mcp.prompt()
def generate_sql(request: str) -> UserMessage:
    """
    Generate a valid SQL query for the user’s request, given the
    current database schema.
    """
    schema = introspect_schema(DB_PATH)
    instruction = (
        f"Database schema (JSON):\n{json.dumps(schema, indent=2)}\n\n"
        f"User request: {request}\n\n"
        "Produce a valid SQL query that answers the request."
    )
    return UserMessage(instruction)

# 3️⃣ Tool: execute arbitrary SQL against the trusted database
@mcp.tool()
def execute_sql(sql: str) -> str:
    """
    Execute the provided SQL statement on the SQLite database
    and return the fetched rows.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    finally:
        conn.close()
    return str(rows)

# 4️⃣ Run the MCP server (supports STDIO, SSE, HTTP)
if __name__ == "__main__":
    mcp.run()

"""
Storage layer. Every run is reproducible: the claim, the exact post set, the
model checkpoint used, and every per-post prediction are all recorded. Two
runs on the same claim three weeks apart can be diffed with a query instead
of trusting memory.

Schema:
    claims       one row per claim being analysed
    posts        social posts belonging to a claim (text + optional image path)
    runs         one row per analysis run of a claim
    predictions  one row per (run, post) -- the stance call and its confidence
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    post_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id    INTEGER NOT NULL REFERENCES claims(claim_id),
    text        TEXT NOT NULL,
    image_path  TEXT,
    author      TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id          INTEGER NOT NULL REFERENCES claims(claim_id),
    classifier_source TEXT NOT NULL,   -- 'checkpoint:<path>' or 'stub'
    status            TEXT NOT NULL,   -- running | awaiting_review | done | failed
    created_at        TEXT NOT NULL,
    finished_at       TEXT,
    report_json       TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL REFERENCES runs(run_id),
    post_id        INTEGER NOT NULL REFERENCES posts(post_id),
    stance         TEXT NOT NULL,   -- support | oppose | neutral
    confidence     REAL NOT NULL,
    source         TEXT NOT NULL,   -- 'model' or 'human'
    reviewed       INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Post:
    post_id: int
    text: str
    image_path: str | None
    author: str | None


def create_claim(text: str, posts: list[dict]) -> tuple[int, list[Post]]:
    """posts: list of {"text": ..., "image_path": optional, "author": optional}"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO claims (text, created_at) VALUES (?, ?)", (text, _now())
        )
        claim_id = cur.lastrowid
        created: list[Post] = []
        for p in posts:
            cur = conn.execute(
                "INSERT INTO posts (claim_id, text, image_path, author) VALUES (?, ?, ?, ?)",
                (claim_id, p["text"], p.get("image_path"), p.get("author")),
            )
            created.append(Post(cur.lastrowid, p["text"], p.get("image_path"), p.get("author")))
        return claim_id, created


def get_posts(claim_id: int) -> list[Post]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT post_id, text, image_path, author FROM posts WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
        return [Post(r["post_id"], r["text"], r["image_path"], r["author"]) for r in rows]


def create_run(claim_id: int, classifier_source: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO runs (claim_id, classifier_source, status, created_at) "
            "VALUES (?, ?, 'running', ?)",
            (claim_id, classifier_source, _now()),
        )
        return cur.lastrowid


def save_predictions(run_id: int, predictions: list[dict]) -> None:
    """predictions: list of {"post_id", "stance", "confidence", "source"}"""
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO predictions (run_id, post_id, stance, confidence, source) "
            "VALUES (:run_id, :post_id, :stance, :confidence, :source)",
            [{**p, "run_id": run_id} for p in predictions],
        )


def get_predictions(run_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT p.*, po.text as post_text FROM predictions p "
            "JOIN posts po ON po.post_id = p.post_id WHERE p.run_id = ?",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def apply_human_review(run_id: int, post_id: int, stance: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET stance = ?, confidence = 1.0, source = 'human', reviewed = 1 "
            "WHERE run_id = ? AND post_id = ?",
            (stance, run_id, post_id),
        )


def set_run_status(run_id: int, status: str, report: dict | None = None) -> None:
    with get_conn() as conn:
        if report is not None:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, report_json = ? WHERE run_id = ?",
                (status, _now(), json.dumps(report), run_id),
            )
        else:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))


def get_run(run_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def get_claim(claim_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return dict(row) if row else None

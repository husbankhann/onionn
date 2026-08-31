"""
Persistence layer (SQLite for simplicity; swap DATABASE_URL logic for
Postgres in production by replacing this module's connection function).

Schema is designed around the actual pain point in the prompt:
  - multiple procurement centers must produce grades that are COMPARABLE
  - every grade must be traceable back to the exact image + config used,
    so disputes can be re-examined objectively instead of "my word vs yours"
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "/tmp/onion_grading.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS procurement_centers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                location TEXT,
                pixels_per_mm REAL DEFAULT 8.0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS grading_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                config_json TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_code TEXT NOT NULL UNIQUE,
                center_id INTEGER NOT NULL,
                farmer_name TEXT,
                variety TEXT,
                target_size_band TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (center_id) REFERENCES procurement_centers(id)
            );

            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lot_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                config_id INTEGER NOT NULL,
                onion_count INTEGER,
                average_composite_score REAL,
                lot_grade TEXT,
                grade_distribution_json TEXT,
                raw_result_json TEXT NOT NULL,
                inspector_name TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (lot_id) REFERENCES lots(id),
                FOREIGN KEY (config_id) REFERENCES grading_configs(id)
            );

            CREATE TABLE IF NOT EXISTS disputes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER NOT NULL,
                raised_by TEXT,
                reason TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                resolution_notes TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                FOREIGN KEY (assessment_id) REFERENCES assessments(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_lot ON assessments(lot_id);
            CREATE INDEX IF NOT EXISTS idx_disputes_assessment ON disputes(assessment_id);
            """
        )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Procurement centers
# ---------------------------------------------------------------------------
def create_center(name, location=None, pixels_per_mm=8.0):
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO procurement_centers (name, location, pixels_per_mm, created_at) VALUES (?, ?, ?, ?)",
            (name, location, pixels_per_mm, now_iso()),
        )
        return cur.lastrowid


def list_centers():
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM procurement_centers ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_center(center_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM procurement_centers WHERE id=?", (center_id,)).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Grading configs (versioned so old assessments remain auditable even after
# thresholds are retuned)
# ---------------------------------------------------------------------------
def save_config(version, config_dict, activate=True):
    with db_session() as conn:
        if activate:
            conn.execute("UPDATE grading_configs SET is_active = 0")
        cur = conn.execute(
            "INSERT INTO grading_configs (version, config_json, is_active, created_at) VALUES (?, ?, ?, ?)",
            (version, json.dumps(config_dict), 1 if activate else 0, now_iso()),
        )
        return cur.lastrowid


def get_active_config():
    with db_session() as conn:
        row = conn.execute(
            "SELECT * FROM grading_configs WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return {"id": row["id"], "version": row["version"], "config": json.loads(row["config_json"])}
        return None


def get_config_by_id(config_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM grading_configs WHERE id=?", (config_id,)).fetchone()
        if row:
            return {"id": row["id"], "version": row["version"], "config": json.loads(row["config_json"])}
        return None


# ---------------------------------------------------------------------------
# Lots
# ---------------------------------------------------------------------------
def create_lot(lot_code, center_id, farmer_name=None, variety=None, target_size_band=None):
    with db_session() as conn:
        cur = conn.execute(
            """INSERT INTO lots (lot_code, center_id, farmer_name, variety, target_size_band, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (lot_code, center_id, farmer_name, variety, target_size_band, now_iso()),
        )
        return cur.lastrowid


def get_lot_by_code(lot_code):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM lots WHERE lot_code=?", (lot_code,)).fetchone()
        return dict(row) if row else None


def get_lot(lot_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM lots WHERE id=?", (lot_id,)).fetchone()
        return dict(row) if row else None


def list_lots(center_id=None):
    with db_session() as conn:
        if center_id:
            rows = conn.execute(
                "SELECT * FROM lots WHERE center_id=? ORDER BY created_at DESC", (center_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM lots ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------
def save_assessment(lot_id, image_filename, config_id, result_dict, inspector_name=None):
    with db_session() as conn:
        summary = result_dict.get("lot_summary", {})
        cur = conn.execute(
            """INSERT INTO assessments
               (lot_id, image_filename, config_id, onion_count, average_composite_score,
                lot_grade, grade_distribution_json, raw_result_json, inspector_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                lot_id,
                image_filename,
                config_id,
                result_dict.get("onion_count", 0),
                summary.get("average_composite_score"),
                summary.get("lot_grade"),
                json.dumps(summary.get("grade_distribution", {})),
                json.dumps(result_dict),
                inspector_name,
                now_iso(),
            ),
        )
        return cur.lastrowid


def get_assessment(assessment_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["raw_result"] = json.loads(d.pop("raw_result_json"))
        d["grade_distribution"] = json.loads(d.pop("grade_distribution_json") or "{}")
        return d


def list_assessments(lot_id=None, center_id=None):
    with db_session() as conn:
        if lot_id:
            rows = conn.execute(
                "SELECT * FROM assessments WHERE lot_id=? ORDER BY created_at DESC", (lot_id,)
            ).fetchall()
        elif center_id:
            rows = conn.execute(
                """SELECT a.* FROM assessments a
                   JOIN lots l ON a.lot_id = l.id
                   WHERE l.center_id = ?
                   ORDER BY a.created_at DESC""",
                (center_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM assessments ORDER BY created_at DESC LIMIT 200").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["grade_distribution"] = json.loads(d.pop("grade_distribution_json") or "{}")
            d.pop("raw_result_json", None)
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# Disputes — this is the direct answer to "resulting in disputes":
# a formal, auditable channel to contest a grade against the stored
# image + measurements + exact config version used.
# ---------------------------------------------------------------------------
def create_dispute(assessment_id, raised_by, reason):
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO disputes (assessment_id, raised_by, reason, status, created_at) VALUES (?, ?, ?, 'open', ?)",
            (assessment_id, raised_by, reason, now_iso()),
        )
        return cur.lastrowid


def resolve_dispute(dispute_id, resolution_notes, status="resolved"):
    with db_session() as conn:
        conn.execute(
            "UPDATE disputes SET status=?, resolution_notes=?, resolved_at=? WHERE id=?",
            (status, resolution_notes, now_iso(), dispute_id),
        )


def list_disputes(status=None):
    with db_session() as conn:
        if status:
            rows = conn.execute("SELECT * FROM disputes WHERE status=? ORDER BY created_at DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM disputes ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_dispute(dispute_id):
    with db_session() as conn:
        row = conn.execute("SELECT * FROM disputes WHERE id=?", (dispute_id,)).fetchone()
        return dict(row) if row else None

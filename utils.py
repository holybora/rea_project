from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

# Public API
__all__ = [
    "load_properties_json",
    "build_and_populate_db",
    "init_sqlite",
]


# Default dataset path (raw/properties.json relative to this file)
_DEF_DATASET = Path(__file__).resolve().parent / "raw" / "properties.json"


def load_properties_json(json_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Load properties list from JSON.

    Args:
        json_path: Optional path to the JSON file. If None, uses raw/properties.json.
    Returns:
        List of dicts representing properties.
    """
    path = Path(json_path) if json_path else _DEF_DATASET
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("properties.json must contain a list of objects")
    return data


def init_sqlite(db_path: str | Path = ":properties.db") -> sqlite3.Connection:
    """Create SQLite connection and enable sensible pragmas."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def _drop_existing_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # Drop in reverse dependency order
    cur.execute("DROP TABLE IF EXISTS nearby_schools;")
    cur.execute("DROP TABLE IF EXISTS amenities;")
    cur.execute("DROP TABLE IF EXISTS properties;")
    conn.commit()


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create tables mirroring the JSON structure in a normalized way.

    Tables:
      - properties: main fields per listing
      - amenities: one row per amenity string per property
      - nearby_schools: one row per school object per property
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS properties (
            property_id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            property_type TEXT,
            price REAL,
            address TEXT,
            city TEXT,
            state TEXT,
            postal_code TEXT,
            bedrooms INTEGER,
            bathrooms INTEGER,
            square_meters INTEGER,
            agent_id INTEGER,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS amenities (
            property_id INTEGER NOT NULL,
            amenity TEXT NOT NULL,
            PRIMARY KEY (property_id, amenity),
            FOREIGN KEY (property_id) REFERENCES properties(property_id) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nearby_schools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            distance_km REAL,
            FOREIGN KEY (property_id) REFERENCES properties(property_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def _extract_scalar(record: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = record.get(key, default)
    # SQLite understands None, int, float, str; convert bools to int for clarity
    if isinstance(val, bool):
        return int(val)
    return val


def _insert_property(cur: sqlite3.Cursor, record: Dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO properties (
            property_id, title, description, property_type, price, address, city, state,
            postal_code, bedrooms, bathrooms, square_meters, agent_id, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(property_id) DO UPDATE SET
            title=excluded.title,
            description=excluded.description,
            property_type=excluded.property_type,
            price=excluded.price,
            address=excluded.address,
            city=excluded.city,
            state=excluded.state,
            postal_code=excluded.postal_code,
            bedrooms=excluded.bedrooms,
            bathrooms=excluded.bathrooms,
            square_meters=excluded.square_meters,
            agent_id=excluded.agent_id,
            status=excluded.status,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at
        ;
        """,
        (
            _extract_scalar(record, "property_id"),
            _extract_scalar(record, "title"),
            _extract_scalar(record, "description"),
            _extract_scalar(record, "property_type"),
            _extract_scalar(record, "price"),
            _extract_scalar(record, "address"),
            _extract_scalar(record, "city"),
            _extract_scalar(record, "state"),
            _extract_scalar(record, "postal_code"),
            _extract_scalar(record, "bedrooms"),
            _extract_scalar(record, "bathrooms"),
            _extract_scalar(record, "square_meters"),
            _extract_scalar(record, "agent_id"),
            _extract_scalar(record, "status"),
            _extract_scalar(record, "created_at"),
            _extract_scalar(record, "updated_at"),
        ),
    )


def _insert_amenities(cur: sqlite3.Cursor, property_id: int, record: Dict[str, Any]) -> None:
    amenities = record.get("amenities") or []
    if isinstance(amenities, list):
        for a in amenities:
            cur.execute(
                """
                INSERT OR IGNORE INTO amenities (property_id, amenity) VALUES (?, ?);
                """,
                (property_id, str(a)),
            )


def _insert_schools(cur: sqlite3.Cursor, property_id: int, record: Dict[str, Any]) -> None:
    schools = record.get("nearby_schools") or []
    if isinstance(schools, list):
        for s in schools:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            distance = s.get("distance_km")
            cur.execute(
                """
                INSERT INTO nearby_schools (property_id, name, distance_km)
                VALUES (?, ?, ?);
                """,
                (property_id, name, distance),
            )


def populate_db(conn: sqlite3.Connection, data: Iterable[Dict[str, Any]]) -> None:
    cur = conn.cursor()
    for rec in data:
        pid = rec.get("property_id")
        if pid is None:
            # Skip malformed rows without primary key
            continue
        _insert_property(cur, rec)
        _insert_amenities(cur, int(pid), rec)
        _insert_schools(cur, int(pid), rec)
    conn.commit()


def build_and_populate_db(
    json_path: Optional[str | Path] = None,
    db_path: str | Path = "properties.db",
    *,
    overwrite: bool = True,
) -> Tuple[sqlite3.Connection, Path]:
    """Build a SQLite database schema mirroring the dataset and load all records.

    Args:
        json_path: path to the dataset JSON. Defaults to raw/properties.json.
        db_path: target SQLite database path (or ":memory:"). If a filesystem
                 path is given and overwrite=True, existing tables are dropped
                 and recreated. If overwrite=False, tables are created if missing
                 and rows upserted.
        overwrite: whether to drop existing tables before creation.
    Returns:
        (connection, db_path as Path)
    """
    data = load_properties_json(json_path)
    conn = init_sqlite(db_path)
    if overwrite:
        _drop_existing_tables(conn)
    _create_schema(conn)
    populate_db(conn, data)
    return conn, Path(db_path)

def main() -> int:
    """CLI entry: build the DB and report the location, then exit 0."""
    conn, db_path = build_and_populate_db()
    print(
        f"Database created and populated at {db_path} (tables: properties, amenities, nearby_schools)."
    )
    try:
        conn.close()
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

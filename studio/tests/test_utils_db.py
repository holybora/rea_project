from pathlib import Path
import sqlite3

from utils import load_properties_json, build_and_populate_db


def test_build_and_populate_db_properties_count(tmp_path: Path):
    # Load dataset via utility
    data = load_properties_json()
    assert isinstance(data, list)
    assert len(data) >= 1

    db_file = tmp_path / "properties.db"
    conn, db_path = build_and_populate_db(db_path=db_file)
    assert db_path.exists()

    cur = conn.cursor()
    # Properties count should match dataset length
    cur.execute("SELECT COUNT(*) FROM properties;")
    (count_props,) = cur.fetchone()
    assert count_props == len(data)

    # There should be at least as many amenities rows as the sum over dataset
    expected_amenities = sum(len(item.get("amenities", []) or []) for item in data)
    cur.execute("SELECT COUNT(*) FROM amenities;")
    (amenities_count,) = cur.fetchone()
    assert amenities_count >= expected_amenities

    # There should be at least as many schools as in dataset
    expected_schools = 0
    for item in data:
        schools = item.get("nearby_schools") or []
        expected_schools += sum(1 for s in schools if isinstance(s, dict))
    cur.execute("SELECT COUNT(*) FROM nearby_schools;")
    (schools_count,) = cur.fetchone()
    assert schools_count >= expected_schools

    conn.close()


def test_specific_values_present(tmp_path: Path):
    # Build DB
    db_file = tmp_path / "props.db"
    conn, _ = build_and_populate_db(db_path=db_file)
    cur = conn.cursor()

    # Verify a known city exists
    cur.execute("SELECT COUNT(*) FROM properties WHERE city = ?;", ("Valencia",))
    (n_valencia,) = cur.fetchone()
    assert n_valencia >= 1

    # Verify a known amenity for a specific property id (121 has wifi & furnished)
    cur.execute(
        "SELECT COUNT(*) FROM amenities WHERE property_id = ? AND amenity = ?;",
        (121, "wifi"),
    )
    (wifi_count,) = cur.fetchone()
    assert wifi_count == 1

    # Verify nearby school recorded for property 121
    cur.execute(
        "SELECT name, distance_km FROM nearby_schools WHERE property_id = ?;",
        (121,),
    )
    rows = cur.fetchall()
    assert any("Universitat de València" in (name or "") for name, _ in rows)
    assert any(isinstance(dist, float) or isinstance(dist, int) for _, dist in rows)

    conn.close()

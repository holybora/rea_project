import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tools.json_search_tool import JSONSearchTool


@pytest.fixture()
def sample_data():
    return [
        {"id": 1, "title": "Cozy Loft", "city": "Valencia", "price": 1200},
        {"id": 2, "title": "Family House", "city": "Paterna", "price": 450000},
        {"id": 3, "title": "Beach Apartment", "city": "Alboraya", "price": 750},
    ]


@pytest.fixture()
def json_file(tmp_path: Path, sample_data):
    p = tmp_path / "data.json"
    p.write_text(json.dumps(sample_data), encoding="utf-8")
    return p


def test_run_basic_search(json_file: Path):
    tool = JSONSearchTool(json_path=str(json_file))

    # Should match across any value, case-insensitive
    results = tool._run("valencia")
    assert len(results) == 1
    assert results[0]["title"] == "Cozy Loft"

    # Substring match in title
    results = tool._run("apartment")
    assert len(results) == 1
    assert results[0]["id"] == 3

    # No match
    results = tool._run("not present")
    assert results == []


def test_run_is_case_insensitive(json_file: Path):
    tool = JSONSearchTool(json_path=str(json_file))
    assert tool._run("VALENCIA")[0]["city"] == "Valencia"


def test_cache_reload_on_file_change(tmp_path: Path, sample_data):
    # Initial file
    p = tmp_path / "data.json"
    p.write_text(json.dumps(sample_data), encoding="utf-8")

    tool = JSONSearchTool(json_path=str(p))

    # First load caches data
    first = tool._run("valencia")
    assert len(first) == 1
    assert tool._last_mtime is not None

    # Modify file with new content and ensure mtime changes
    new_data = [
        {"id": 10, "title": "New Listing", "city": "Gandia", "price": 999},
    ]
    # Sleep to ensure filesystem mtime differs on platforms with 1-second resolution
    time.sleep(1.1)
    p.write_text(json.dumps(new_data), encoding="utf-8")

    # Confirm OS mtime has changed
    assert os.path.getmtime(p) != tool._last_mtime

    # Next run should reload and reflect new data
    after = tool._run("gandia")
    assert len(after) == 1
    assert after[0]["id"] == 10


@pytest.mark.asyncio
async def test_arun_matches_run(json_file: Path):
    tool = JSONSearchTool(json_path=str(json_file))
    sync_results = tool._run("house")
    async_results = await tool._arun("house")
    assert async_results == sync_results


# Additional tests using the actual dataset in raw/properties.json
from pathlib import Path as _Path


def _real_dataset_path() -> _Path:
    # tests/ is one level below project root
    return _Path(__file__).resolve().parents[1] / "raw" / "properties.json"


def test_search_against_actual_dataset_unique_and_multiple():
    dataset = _real_dataset_path()
    assert dataset.exists(), f"Dataset not found at {dataset}"

    tool = JSONSearchTool(json_path=str(dataset))

    # Unique-ish term in title
    res_penthouse = tool._run("Penthouse")
    assert any(r.get("title", "").lower().find("penthouse") >= 0 for r in res_penthouse)
    assert len(res_penthouse) >= 1

    # City with multiple listings in dataset
    res_valencia = tool._run("Valencia")
    assert len(res_valencia) >= 1
    assert any(r.get("city") == "Valencia" for r in res_valencia)


def test_search_amenity_wifi_in_actual_dataset():
    dataset = _real_dataset_path()
    tool = JSONSearchTool(json_path=str(dataset))

    res = tool._run("wifi")
    assert len(res) >= 1
    # At least one result should have wifi in amenities
    assert any("wifi" in [str(v).lower() for v in item.get("amenities", [])] or "wifi" in str(item).lower() for item in res)


def test_empty_query_behavior_on_actual_dataset():
    dataset = _real_dataset_path()
    tool = JSONSearchTool(json_path=str(dataset))

    # Current behavior: empty string matches everything (substring of all strings)
    all_results = tool._run("")
    assert isinstance(all_results, list)
    assert len(all_results) >= 1  # Documenting current behavior

    # Whitespace-only query should still return some matches (likely many titles/addresses contain spaces)
    space_results = tool._run(" ")
    assert len(space_results) >= 1


def test_non_string_query_raises_typeerror():
    dataset = _real_dataset_path()
    tool = JSONSearchTool(json_path=str(dataset))

    with pytest.raises(Exception):
        tool._run(None)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        tool._run(123)  # type: ignore[arg-type]
    with pytest.raises(Exception):
        tool._run(["list"])  # type: ignore[arg-type]


def test_no_results_for_unlikely_term_on_actual_dataset():
    dataset = _real_dataset_path()
    tool = JSONSearchTool(json_path=str(dataset))

    results = tool._run("zzzzunlikelysearchterm12345")
    assert results == []

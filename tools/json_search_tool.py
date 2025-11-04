import json
import os
import unicodedata
from typing import List, Dict, Any, Optional

from pydantic import PrivateAttr
from langchain.tools import BaseTool


def _normalize(text: str) -> str:
    """Return a case- and accent-insensitive representation of the text.

    This improves multilingual matching by removing diacritics and normalizing
    Unicode so that, e.g., "València" matches "Valencia" and Cyrillic/Greek
    characters compare consistently when appropriate.
    """
    # Normalize to NFKD and strip combining marks (diacritics), then lowercase
    nfkd = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return without_accents.casefold()


class JSONSearchTool(BaseTool):
    name: str = "property_search"
    description: str = (
        "Search available properties by any criteria (location, type, price, amenities, etc.). "
        "Returns matching property listings from the dataset. Matching is case- and accent-insensitive."
    )

    # Public field for configuring the tool
    json_path: str

    # Private runtime cache (not part of the model schema)
    _cache: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    _last_mtime: Optional[float] = PrivateAttr(default=None)

    def _load_json(self) -> List[Dict[str, Any]]:
        """Load and cache JSON file; reload only if file changes."""
        mtime = os.path.getmtime(self.json_path)
        if self._cache is None or self._last_mtime != mtime:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
            self._last_mtime = mtime
        return self._cache or []

    def _run(self, query: str) -> List[Dict[str, Any]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        data = self._load_json()
        q = _normalize(query)
        results: List[Dict[str, Any]] = []
        for item in data:
            # Compare against all top-level values stringified with normalization
            if any(q in _normalize(str(v)) for v in item.values()):
                results.append(item)
        return results

    async def _arun(self, query: str) -> List[Dict[str, Any]]:
        # async version
        return self._run(query)
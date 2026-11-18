"""OSIRIS Notebook file format (``*.dnanb``), version 1.

A notebook is an ordinary project file (stored, versioned and access-controlled by the editor
store) whose content is canonical JSON:

    {"format": "osiris-notebook", "version": 1, "title": "...",
     "cells": [{"id": "c1", "type": "dnalang" | "markdown", "source": "..."}, ...]}

Only ``dnalang`` cells execute. A cell run is bound to the notebook file revision it was read
from and to the digest of that cell's source, so a result can be traced to exactly one cell
of exactly one saved revision. The web client writes the same format (lib/osiris/notebook-model.ts).
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict

FORMAT = "osiris-notebook"
VERSION = 1
CELL_TYPES = ("dnalang", "markdown")
EXECUTABLE = ("dnalang",)
MAX_CELLS = 200
_CELL_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class NotebookFormatError(ValueError):
    pass


def parse(text: str) -> Dict[str, Any]:
    try:
        nb = json.loads(text)
    except (TypeError, ValueError):
        raise NotebookFormatError("not an OSIRIS notebook: content is not JSON")
    if not isinstance(nb, dict) or nb.get("format") != FORMAT:
        raise NotebookFormatError(f"not an OSIRIS notebook: format must be {FORMAT!r}")
    if nb.get("version") != VERSION:
        raise NotebookFormatError(f"unsupported notebook version {nb.get('version')!r} (this service reads {VERSION})")
    cells = nb.get("cells")
    if not isinstance(cells, list) or len(cells) > MAX_CELLS:
        raise NotebookFormatError(f"cells must be a list of at most {MAX_CELLS}")
    seen = set()
    for c in cells:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not _CELL_ID.match(c["id"]):
            raise NotebookFormatError("every cell needs an id of 1-40 letters, digits, '_' or '-'")
        if c["id"] in seen:
            raise NotebookFormatError(f"duplicate cell id {c['id']!r}")
        seen.add(c["id"])
        if c.get("type") not in CELL_TYPES:
            raise NotebookFormatError(f"cell {c['id']}: type must be one of {CELL_TYPES}")
        if not isinstance(c.get("source"), str):
            raise NotebookFormatError(f"cell {c['id']}: source must be text")
    return nb


def cell(nb: Dict[str, Any], cell_id: Any) -> Dict[str, Any]:
    for c in nb["cells"]:
        if c["id"] == cell_id:
            return c
    raise KeyError(cell_id)


def source_sha256(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()

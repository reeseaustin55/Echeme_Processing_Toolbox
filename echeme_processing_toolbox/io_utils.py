"""Utility functions for reading Biologic electrochemistry data files."""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .dependencies import np

try:  # Optional dependency for .mpr parsing
    from eclabfiles import MPRfile  # type: ignore
    _MPR_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional import guard
    MPRfile = None
    _MPR_IMPORT_ERROR = exc
except Exception as exc:  # pragma: no cover - unexpected import failure
    MPRfile = None
    _MPR_IMPORT_ERROR = exc


_HEADER_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class NumericTable:
    """Container for tabular numeric data with labelled columns."""

    columns: Dict[str, np.ndarray]

    def get(self, *candidates: str) -> np.ndarray:
        """Return the first available column among ``candidates``."""

        for name in candidates:
            if name in self.columns:
                return self.columns[name]
        raise KeyError(f"None of the requested columns are available: {', '.join(candidates)}")

    def require(self, *names: str) -> List[np.ndarray]:
        """Return columns by exact name."""

        return [self.get(name) for name in names]


def _sanitize_header_name(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("<", "").replace(">", "")
    name = name.replace("omeg", "omega")  # common typo guard
    name = _HEADER_SANITIZE_RE.sub("_", name)
    name = name.strip("_")
    return name


def read_numeric_table(path: Path) -> NumericTable:
    """Read a tab-separated text file exported from Biologic instruments.

    The reader is intentionally permissive – it skips non-numeric rows and
    attempts to infer column labels from the first header-like row.  When no
    header row is present, the columns are indexed numerically (``col0``,
    ``col1``, ...).
    """

    columns: Dict[str, List[float]] = {}
    header: Optional[List[str]] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw_row in reader:
            if not raw_row:
                continue
            row = [item.strip() for item in raw_row]
            # Detect header: contains any alphabetic character or slash.
            if header is None and any(re.search(r"[A-Za-z]", item) for item in row):
                header = [_sanitize_header_name(item) or f"col{idx}" for idx, item in enumerate(row)]
                for name in header:
                    columns.setdefault(name, [])
                continue

            # Attempt to parse numbers; skip row on failure.
            numeric_row: List[float] = []
            valid_row = True
            for item in row:
                if item == "":
                    numeric_row.append(math.nan)
                    continue
                try:
                    numeric_row.append(float(item))
                except ValueError:
                    valid_row = False
                    break
            if not valid_row:
                continue

            # Lazily create default headers when missing.
            if header is None:
                header = [f"col{idx}" for idx in range(len(numeric_row))]
                for name in header:
                    columns.setdefault(name, [])
            elif len(numeric_row) != len(header):
                # Skip inconsistent row lengths.
                continue

            for name, value in zip(header, numeric_row):
                columns[name].append(value)

    if not columns:
        raise ValueError(f"No numeric data could be parsed from '{path}'.")

    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    return NumericTable(arrays)


def read_mpr_table(path: Path) -> NumericTable:
    """Load data from a Biologic ``.mpr`` binary file if support is available."""

    if MPRfile is None:
        extra = f" (original error: {_MPR_IMPORT_ERROR})" if _MPR_IMPORT_ERROR else ""
        raise RuntimeError(
            "Reading .mpr files requires the optional 'eclabfiles' package and its dependencies. "
            "Install it with 'pip install eclabfiles' or export the data as .txt." + extra
        )

    mpr = MPRfile(str(path))  # type: ignore[call-arg]
    if hasattr(mpr, "data"):
        data = mpr.data  # pandas.DataFrame like
        columns = { _sanitize_header_name(str(col)): np.asarray(data[col].values, dtype=float) for col in data.columns }
        return NumericTable(columns)

    # Fallback: build dict from lists when .data not available (older versions)
    columns = { _sanitize_header_name(key): np.asarray(values, dtype=float) for key, values in mpr.items() }  # type: ignore[attr-defined]
    return NumericTable(columns)


def load_table(path: Path) -> NumericTable:
    """Dispatch loader based on file extension."""

    ext = path.suffix.lower()
    if ext == ".mpr":
        return read_mpr_table(path)
    return read_numeric_table(path)


def iter_files(folder: Path, extensions: Iterable[str]) -> Iterable[Path]:
    """Yield files with matching extensions within ``folder`` (non-recursive)."""

    lowered = {ext.lower() for ext in extensions}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in lowered:
            yield path

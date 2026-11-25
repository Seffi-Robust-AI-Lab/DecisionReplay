"""Select the model-by-benchmark rectangle and name its models for a corpus tier."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from .schema import TABLES_BY_TIER


def _cleared_cells(clearance, catalog_ids, routes):
    """Return cleared pairs and the digest of the clearance records used.

    A file with a ``release`` column (the release matrix of ``analysis.release``) selects the cells
    marked ``released``; a file without it (the review's own matrix) selects those whose
    ``numeric_public_status`` is ``cleared``.
    """
    if clearance is None:
        raise ValueError("the open tier requires a clearance file")
    required = {"catalog_id", "route_id", "numeric_public_status"}
    seen = set()
    cleared = set()
    data = Path(clearance).read_bytes()
    with io.StringIO(data.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not required.issubset(columns) or len(set(columns)) != len(columns):
            raise ValueError("invalid columns in the clearance file")
        by_release = "release" in columns
        for row in reader:
            if None in row or None in row.values():
                raise ValueError("incomplete record in the clearance file")
            try:
                cid = int(row["catalog_id"])
            except (TypeError, ValueError):
                raise ValueError("invalid catalogue id in the clearance file") from None
            route = row["route_id"]
            if cid not in catalog_ids or route not in routes:
                raise ValueError("clearance names a cell outside the source export")
            cell = (cid, route)
            if cell in seen:
                raise ValueError("duplicate cell in the clearance file")
            seen.add(cell)
            if by_release:
                if row["release"] not in ("released", "held"):
                    raise ValueError("invalid release value in the clearance file")
                if row["release"] == "released":
                    cleared.add(cell)
            elif row["numeric_public_status"] == "cleared":
                cleared.add(cell)
    return cleared, hashlib.sha256(data).hexdigest()


def select_cells(tier, benchmarks, models, clearance=None):
    """Return catalogue ids and model indices in source order, plus the open clearance digest."""
    if tier not in TABLES_BY_TIER:
        raise ValueError("unknown corpus tier")
    catalog_ids = [int(row["catalog_id"]) for row in benchmarks]
    indices = [int(row["model_index"]) for row in models]
    routes = [row["route_id"] for row in models]
    if any(cid < 0 for cid in catalog_ids) or len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("invalid or duplicate catalogue id in benchmarks.csv")
    if indices != list(range(len(models))):
        raise ValueError("models.csv must be in model index order")
    if tier != "open":
        return catalog_ids, indices, None
    if len(set(routes)) != len(routes):
        raise ValueError("open clearance requires unique route ids in models.csv")

    cleared, digest = _cleared_cells(clearance, set(catalog_ids), set(routes))
    cleared_routes = {route for cid, route in cleared}
    selected = [index for index in indices if routes[index] in cleared_routes]
    common = [cid for cid in catalog_ids if all((cid, route) in cleared for route in cleared_routes)]
    if not selected or not common:
        raise ValueError("open clearance has no nonempty rectangle across the cleared models")
    return common, selected, digest


def _code(value, codes, prefix):
    """Return a stable code assigned by first appearance, preserving an empty value."""
    if not value:
        return ""
    if value not in codes:
        codes[value] = f"{prefix}{len(codes) + 1:02d}"
    return codes[value]


def model_rows(models, mmeta, selected, names):
    """Return models.csv records with tier indices and original model and group codes."""
    if names not in ("real", "coded"):
        raise ValueError("unknown names mode")
    organisations = {}
    bases = {}
    output = []
    for index, model in enumerate(models):
        meta = mmeta[index]
        model_code = f"M{index:02d}"
        name = model["route_id"]
        organisation = meta["organisation"]
        base = meta["base_model"]
        if names == "coded":
            name = model_code
            organisation = _code(organisation, organisations, "O")
            base = _code(base, bases, "B")
        if index in selected:
            output.append(dict(model_index=len(output), model_code=model_code, model_name=name,
                               organisation=organisation, base_model=base,
                               input_price_usd_per_token=meta["input_price_usd_per_token"]))
    return output

#!/usr/bin/env python3
"""Generate exact-profile rule-source catalogs from the governed Saudi base.

The public clauses and ownership bindings are shared.  Each published rules
profile still receives an exact retained-source record and an immutable catalog
identity, so a certification profile cannot be substituted across variants.
"""

from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from baloot_platform.core.rules import RuleBook  # noqa: E402
from baloot_platform.filesystem import atomic_write_text  # noqa: E402


BASE = ROOT / "configs/rules/source_catalog_v1.yaml"
_BASE_CONFIG_SOURCE = "balootx_configured_gap_2026"
_VARIANTS = (
    {
        "profile": "saudi_federation_2023_v4.yaml",
        "catalog_id": "saudi_federation_2023_v4_sources",
        "catalog_status": "public_regulation_plus_unratified_configured_gaps",
        "source_id": "balootx_saudi_federation_2023_v4_config",
        "output": "source_catalog_saudi_federation_2023_v4.yaml",
    },
    {
        "profile": "saudi_federation_2023_v3.yaml",
        "catalog_id": "saudi_federation_2023_v3_sources",
        "catalog_status": "public_regulation_plus_unratified_configured_gaps",
        "source_id": "balootx_saudi_federation_2023_v3_config",
        "output": "source_catalog_saudi_federation_2023_v3.yaml",
    },
    {
        "profile": "saudi_federation_2023_v2.yaml",
        "catalog_id": "saudi_federation_2023_v2_sources",
        "catalog_status": "public_regulation_plus_unratified_configured_gaps",
        "source_id": "balootx_saudi_federation_2023_v2_config",
        "output": "source_catalog_saudi_federation_2023_v2.yaml",
    },
    {
        "profile": "saudi_project_optional_unratified_v1.yaml",
        "catalog_id": "saudi_project_optional_unratified_v1_sources",
        "catalog_status": "unratified_configured_variant",
        "source_id": "balootx_saudi_project_optional_unratified_v1_config",
        "output": "source_catalog_saudi_project_optional_unratified_v1.yaml",
    },
)


def _replace_source_id(value: Any, source_id: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_source_id(item, source_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_source_id(item, source_id) for item in value]
    return source_id if value == _BASE_CONFIG_SOURCE else value


def build_variant(spec: dict[str, str]) -> dict[str, Any]:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    profile_path = ROOT / "configs/rules" / spec["profile"]
    rules = RuleBook.load(profile_path)
    value = _replace_source_id(deepcopy(base), spec["source_id"])
    value["catalog"] = {
        "id": spec["catalog_id"],
        "version": "1.0.0",
        "profile_id": rules.id,
        "profile_version": rules.version,
        "profile_sha256": rules.sha256,
        "certification_status": spec["catalog_status"],
    }
    sources = [
        source
        for source in value["sources"]
        if source["id"] != spec["source_id"]
    ]
    sources.append(
        {
            "id": spec["source_id"],
            "authority_class": "configured_variant",
            "url": f"repo://configs/rules/{spec['profile']}",
            "snapshot_url": f"repo://configs/rules/{spec['profile']}",
            "content_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
            "published_date": "2026-08-31",
            "status": "internal_unratified",
        }
    )
    value["sources"] = sources
    value["bindings"].append(
        {
            "path_prefix": "projects.declaration_mode",
            "authority_class": "configured_variant",
            "source_ids": [spec["source_id"]],
            "clause_locator": "configured_profile.projects.declaration_mode",
            "clause": "mandatory or optional project-declaration transition policy",
            "applicability": "project declaration actions",
            "examples": ["tests/test_optional_project_declarations_v1.py"],
            "counterexamples": [
                "mandatory profile rejects pass while optional profile permits pass"
            ],
            "runtime_owner": "src/baloot_platform/core/project_ir.py",
            "cp_sat_owner": "src/baloot_platform/research/ortools_project_model.py",
            "ratification_status": "external_referee_required",
        }
    )
    if rules.data["projects"].get("top_rank_visibility") == "comparison_only":
        value["bindings"][-1]["ratification_status"] = "pending_owner_ratification"
        value["bindings"].append({
            "path_prefix": "projects.top_rank_visibility",
            "authority_class": "configured_variant",
            "source_ids": [spec["source_id"]],
            "clause_locator": "configured_profile.projects.top_rank_visibility",
            "clause": "opponent top rank is hidden unless needed for project comparison",
            "applicability": "observer projections and consistent hidden worlds",
            "examples": ["tests/test_observer_information_projection.py"],
            "counterexamples": ["losing project category does not expose its top rank"],
            "runtime_owner": "src/baloot_platform/core/project_visibility.py",
            "cp_sat_owner": "src/baloot_platform/research/ortools_information_set_compiler.py",
            "ratification_status": "pending_owner_ratification",
        })
    return value


def _text(value: dict[str, Any]) -> str:
    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        width=100,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale: list[str] = []
    for spec in _VARIANTS:
        target = ROOT / "configs/rules" / spec["output"]
        expected = _text(build_variant(spec))
        if args.check:
            if not target.is_file() or target.read_text(encoding="utf-8") != expected:
                stale.append(target.relative_to(ROOT).as_posix())
        else:
            atomic_write_text(target, expected)
            print(target)
    if stale:
        raise SystemExit(f"stale generated rule-source catalogs: {', '.join(stale)}")


if __name__ == "__main__":
    main()

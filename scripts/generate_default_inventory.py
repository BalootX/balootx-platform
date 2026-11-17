#!/usr/bin/env python3
"""Inventory active constructor and dataclass defaults."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/architecture/dataclass-default-inventory.json"
PACKAGES = ("configuration", "core", "analysis", "video", "plugins")
GOVERNED_LEGACY_REQUESTS = {"ScenarioRequest", "MatchScenarioRequest"}
SAFETY_CLASSES = {"PluginDescriptor"}


def build() -> dict[str, object]:
    entries: list[dict[str, object]] = []
    source_root = ROOT / "src/baloot_platform"
    for package in PACKAGES:
        for path in sorted((source_root / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                decorators = {ast.unparse(item).split("(", 1)[0] for item in node.decorator_list}
                if "dataclass" not in decorators:
                    continue
                for field in node.body:
                    if not isinstance(field, ast.AnnAssign) or field.value is None:
                        continue
                    field_name = getattr(field.target, "id", None)
                    if not field_name:
                        continue
                    if node.name in GOVERNED_LEGACY_REQUESTS:
                        classification = "governed_tuning"
                        status = "migration_required"
                        owner = "configs/challenge/progressive_mining_v1.yaml"
                        rationale = (
                            "Legacy request compatibility default; runtime callers "
                            "must migrate to an explicit resolved profile."
                        )
                    elif node.name in SAFETY_CLASSES:
                        classification = "safety_bound"
                        status = "compliant"
                        owner = "schemas/plugin-manifest.schema.json"
                        rationale = (
                            "Manifest protocol or deny-by-default safety value; "
                            "registry policy remains explicit and governed."
                        )
                    else:
                        classification = "protocol_or_mathematical_invariant"
                        status = "compliant"
                        owner = f"{path.relative_to(ROOT).as_posix()}:{node.name}"
                        rationale = (
                            "Neutral serialized state, result, or absence value; "
                            "it does not select product tuning."
                        )
                    entries.append(
                        {
                            "key": (
                                f"{path.relative_to(ROOT).as_posix()}:"
                                f"{node.name}.{field_name}"
                            ),
                            "default": ast.unparse(field.value),
                            "classification": classification,
                            "status": status,
                            "owner": owner,
                            "rationale": rationale,
                        }
                    )
    entries.sort(key=lambda item: str(item["key"]))
    return {
        "schemaVersion": 1,
        "version": "21.0.0",
        "scope": {
            "kind": "dataclass-field-defaults",
            "packages": list(PACKAGES),
        },
        "entries": entries,
        "summary": {
            "total": len(entries),
            "migrationRequired": sum(
                item["status"] == "migration_required" for item in entries
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("dataclass default inventory is stale")
        return 0
    OUTPUT.write_text(rendered, encoding="utf-8", newline="\n")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

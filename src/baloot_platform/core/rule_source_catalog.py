from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from .rules import RuleBook


_BINDING_FIELDS = {
    "path_prefix", "authority_class", "source_ids", "clause_locator", "clause", "applicability",
    "examples", "counterexamples", "runtime_owner", "cp_sat_owner",
    "ratification_status",
}


def _leaves(value: Any, prefix: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.extend(_leaves(value[key], path))
        return tuple(result)
    return (prefix,)


class RuleSourceCatalogV1:
    """Closed clause/source ownership for every leaf in one frozen rule profile."""

    def __init__(self, data: Mapping[str, Any], rules: RuleBook):
        self._data = copy.deepcopy(dict(data))
        if set(self._data) != {"schema_version", "catalog", "sources", "bindings"}:
            raise ValueError("rule source catalog has missing or unknown sections")
        if self._data["schema_version"] != 1:
            raise ValueError("unsupported rule source catalog schema")
        catalog = dict(self._data["catalog"])
        expected = {
            "id", "version", "profile_id", "profile_version", "profile_sha256",
            "certification_status",
        }
        if set(catalog) != expected:
            raise ValueError("rule source catalog metadata is not closed")
        if (
            catalog["profile_id"] != rules.id
            or catalog["profile_version"] != rules.version
            or catalog["profile_sha256"] != rules.sha256
        ):
            raise ValueError("rule source catalog does not match the exact rules profile")

        sources = tuple(dict(item) for item in self._data["sources"])
        if any(not item.get("id") for item in sources):
            raise ValueError("rule sources require IDs")
        source_ids = {str(item["id"]) for item in sources}
        if len(source_ids) != len(sources):
            raise ValueError("rule sources require unique IDs")
        for source in sources:
            required = {
                "id", "authority_class", "url", "snapshot_url", "content_sha256",
                "published_date", "status",
            }
            digest = source.get("content_sha256")
            if (
                set(source) != required
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            ):
                raise ValueError("rule source metadata is incomplete")

        bindings = tuple(dict(item) for item in self._data["bindings"])
        prefixes = [str(item.get("path_prefix")) for item in bindings]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("rule source binding prefixes must be unique")
        for binding in bindings:
            if set(binding) != _BINDING_FIELDS:
                raise ValueError("rule source binding is incomplete or has unknown fields")
            if not all(binding[key] for key in _BINDING_FIELDS - {"counterexamples"}):
                raise ValueError("rule source binding contains an empty required value")
            unknown = set(binding["source_ids"]) - source_ids
            if unknown:
                raise ValueError(f"rule source binding references unknown sources: {sorted(unknown)}")

        coverage: dict[str, dict[str, Any]] = {}
        for leaf in _leaves(rules.data):
            matches = [
                item for item in bindings
                if leaf == item["path_prefix"] or leaf.startswith(f"{item['path_prefix']}.")
            ]
            if not matches:
                raise ValueError(f"rule leaf has no source binding: {leaf}")
            coverage[leaf] = max(matches, key=lambda item: len(item["path_prefix"]))
        self._sources = sources
        self._bindings = bindings
        self._coverage = coverage
        canonical = json.dumps(self._data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def sources(self) -> tuple[dict[str, Any], ...]:
        return copy.deepcopy(self._sources)

    @property
    def bindings(self) -> tuple[dict[str, Any], ...]:
        return copy.deepcopy(self._bindings)

    @property
    def coverage(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(self._coverage)

    @classmethod
    def load(cls, path: str | Path, rules: RuleBook) -> "RuleSourceCatalogV1":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")), rules)

    def binding_for(self, profile_path: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._coverage[str(profile_path)])
        except KeyError as exc:
            raise KeyError(f"unknown rule profile leaf: {profile_path}") from exc

    def verify_retained_sources(
        self, repository_root: str | Path
    ) -> tuple[dict[str, Any], ...]:
        """Verify byte hashes for sources retained inside the repository.

        External URLs remain explicit ``not_retained`` records; they are never
        treated as locally reproducible evidence merely because the catalog
        records a historical hash.
        """

        root = Path(repository_root).resolve()
        receipts: list[dict[str, Any]] = []
        for source in self._sources:
            source_id = str(source["id"])
            snapshot_url = str(source["snapshot_url"])
            if not snapshot_url.startswith("repo://"):
                receipts.append(
                    {
                        "sourceId": source_id,
                        "status": "not_retained",
                        "expectedSha256": str(source["content_sha256"]),
                    }
                )
                continue
            path = (root / snapshot_url.removeprefix("repo://")).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"rule source escapes repository root: {source_id}")
            if not path.is_file():
                raise ValueError(f"retained rule source is missing: {source_id}")
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            expected = str(source["content_sha256"])
            if actual != expected.lower():
                raise ValueError(f"retained rule source hash mismatch: {source_id}")
            receipts.append(
                {
                    "sourceId": source_id,
                    "status": "verified",
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(payload),
                    "expectedSha256": expected,
                    "actualSha256": actual,
                }
            )
        return tuple(receipts)


def load_bound_rule_source_catalog(
    certification: Mapping[str, Any],
    rules: RuleBook,
    repository_root: str | Path,
) -> RuleSourceCatalogV1 | None:
    """Load the catalog named by a certification profile and verify its identity."""

    binding = certification.get("rule_source_catalog")
    if binding is None:
        return None
    binding = dict(binding)
    if set(binding) != {"path", "id", "version", "sha256"}:
        raise ValueError("certification rule source catalog binding is not closed")
    root = Path(repository_root).resolve()
    path = (root / "configs" / str(binding["path"])).resolve()
    if not path.is_relative_to(root / "configs"):
        raise ValueError("certification rule source catalog escapes configuration root")
    catalog = RuleSourceCatalogV1.load(path, rules)
    identity = dict(catalog.data["catalog"])
    if (
        str(binding["id"]) != str(identity["id"])
        or str(binding["version"]) != str(identity["version"])
        or str(binding["sha256"]) != catalog.sha256
    ):
        raise ValueError("certification rule source catalog identity is stale")
    catalog.verify_retained_sources(root)
    return catalog


def load_exact_rule_source_catalog(
    rules: RuleBook,
    repository_root: str | Path,
) -> RuleSourceCatalogV1:
    """Resolve the single catalog bound to an exact rules-profile identity.

    Native Research workers receive the frozen ``RuleBook`` rather than a
    process-active certification profile.  Discovery is therefore limited to
    catalogs whose closed metadata exactly matches that rulebook; zero or
    multiple distinct matches fail closed.
    """

    root = Path(repository_root).resolve()
    catalogs: dict[str, RuleSourceCatalogV1] = {}
    for path in sorted((root / "configs" / "rules").glob("source_catalog*.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(document, Mapping):
            continue
        identity = document.get("catalog")
        if not isinstance(identity, Mapping):
            continue
        if (
            identity.get("profile_id") != rules.id
            or identity.get("profile_version") != rules.version
            or identity.get("profile_sha256") != rules.sha256
        ):
            continue
        catalog = RuleSourceCatalogV1(document, rules)
        catalogs[catalog.sha256] = catalog
    if not catalogs:
        raise ValueError("exact rules profile has no rule source catalog")
    if len(catalogs) != 1:
        raise ValueError("exact rules profile has ambiguous rule source catalogs")
    catalog = next(iter(catalogs.values()))
    catalog.verify_retained_sources(root)
    return catalog


def validate_certification_rules_identity(
    certification: Mapping[str, Any], rules: RuleBook
) -> None:
    expected = certification.get("candidate_profile_identity")
    if expected is None:
        return
    expected = dict(expected)
    if set(expected) != {"id", "version", "sha256"}:
        raise ValueError("certification rules profile identity is not closed")
    if expected != {
        "id": rules.id,
        "version": rules.version,
        "sha256": rules.sha256,
    }:
        raise ValueError("certification rules profile identity is stale")


__all__ = [
    "RuleSourceCatalogV1",
    "load_bound_rule_source_catalog",
    "load_exact_rule_source_catalog",
    "validate_certification_rules_identity",
]

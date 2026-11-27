from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import yaml
from jsonschema.exceptions import best_match
from referencing import Registry
from referencing.jsonschema import DRAFT202012

from .core.rule_source_catalog import (
    load_bound_rule_source_catalog,
    validate_certification_rules_identity,
)
from .core.rules import RuleBook


_VIDEO_TEMPLATE_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("identity", "enrollment_uri_template"): frozenset({"game_serial"}),
    ("localizations", "video.decision.generated_title"): frozenset({"number"}),
    ("localizations", "video.bid.challenge_title"): frozenset({"number"}),
    ("localizations", "video.analysis.verified_tactical"): frozenset({"best"}),
    ("localizations", "video.analysis.only_winning_move"): frozenset({"best"}),
    ("localizations", "video.analysis.unique_awarded_margin"): frozenset(
        {"selectedMargin", "nextBestMargin"}
    ),
    ("localizations", "video.analysis.contract_rescue"): frozenset({"best"}),
    ("localizations", "video.analysis.only_winning_contract_rescue"): frozenset(
        {"best", "alternative"}
    ),
    ("localizations", "video.analysis.small_card_big_gain"): frozenset({"best"}),
    ("localizations", "video.analysis.large_card_sacrifice"): frozenset({"best"}),
    ("localizations", "video.analysis.kabout_creation_prevention"): frozenset({"best"}),
    ("localizations", "video.analysis.kabout_creation"): frozenset({"best"}),
    ("localizations", "video.analysis.kabout_prevention"): frozenset({"best"}),
    ("localizations", "video.analysis.entry_unblocking"): frozenset({"best"}),
    ("localizations", "video.analysis.trump_control"): frozenset({"best"}),
    ("localizations", "video.analysis.partner_enablement"): frozenset({"best"}),
    ("localizations", "video.purchase.hokum_summary"): frozenset(
        {"contract", "round", "trump"}
    ),
    ("localizations", "video.purchase.plain_summary"): frozenset({"contract", "round"}),
    ("localizations", "video.decision.your_turn"): frozenset({"name"}),
    ("challenge_copy", "buyer_with_name"): frozenset({"buyer"}),
    ("challenge_copy", "purchase_hokum_summary"): frozenset(
        {"contract", "round", "trump"}
    ),
    ("challenge_copy", "purchase_plain_summary"): frozenset({"contract", "round"}),
    ("challenge_copy", "metric_round_margin"): frozenset({"value"}),
    ("challenge_copy", "metric_contract_success"): frozenset({"value"}),
    ("challenge_copy", "metric_round_lead"): frozenset({"value"}),
    ("challenge_copy", "request_error"): frozenset({"status"}),
    ("challenge_copy", "action_accessibility"): frozenset({"action"}),
    ("challenge_copy", "action_accessibility_en"): frozenset({"action"}),
    ("challenge_copy", "action_project_declare"): frozenset({"project"}),
    ("challenge_copy", "action_project_declare_en"): frozenset({"project"}),
    ("challenge_copy", "player_path_first"): frozenset({"choice"}),
    ("challenge_copy", "player_path_second"): frozenset({"choice"}),
    ("challenge_copy", "player_seat_summary"): frozenset({"seat"}),
    ("challenge_copy", "player_start_score"): frozenset({"score"}),
    ("challenge_copy", "player_purchase_card"): frozenset({"card"}),
    ("challenge_copy", "player_turn_named"): frozenset({"player"}),
    ("challenge_copy", "player_bid_hokum"): frozenset({"suit"}),
    ("challenge_copy", "player_result_body"): frozenset({"rounds"}),
    ("challenge_copy", "player_share_created"): frozenset({"code"}),
    ("challenge_copy", "player_replay_verified"): frozenset({"actions"}),
    ("challenge_copy", "player_replay_failed"): frozenset({"issues"}),
}
_TEMPLATE_FIELD = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def _validate_video_copy_templates(data: Mapping[str, Any]) -> None:
    """Keep editable copy compatible with every Python and browser consumer."""
    for (section, key), expected in _VIDEO_TEMPLATE_FIELDS.items():
        if key == "video.analysis.unique_awarded_margin" and key not in dict(data.get(section) or {}):
            continue  # Historical video profiles predate score-swing explanations.
        if key == "video.decision.your_turn" and "majlis" not in data:
            continue  # Historical video profiles predate the Majlis look.
        value = str(dict(data.get(section) or {}).get(key) or "")
        fields = frozenset(_TEMPLATE_FIELD.findall(value))
        residue = _TEMPLATE_FIELD.sub("", value)
        if "{" in residue or "}" in residue or fields != expected:
            dotted = f"{section}.{key}"
            raise ValueError(
                f"video copy template {dotted} must contain exactly "
                f"{sorted(expected)} placeholders"
            )


def _verified_inherited_cpu_ceiling(checkout: Path) -> bool:
    """Avoid recompiling the launcher only when Windows confirms its live limits."""
    import ctypes
    import os
    from ctypes import wintypes

    name = os.environ.get("BALOOT_CPU_JOB_NAME", "")
    if not name:
        return False
    try:
        config = json.loads((checkout / "configs/runtime/local_cpu.json").read_text())
        if (
            not isinstance(config, dict)
            or set(config) != {"schemaVersion", "cpuPercent"}
            or type(config["schemaVersion"]) is not int
            or config["schemaVersion"] != 1
            or type(config["cpuPercent"]) is not int
            or not 1 <= config["cpuPercent"] <= 25
        ):
            return False
        logical_cpus = os.cpu_count() or 0
        count = logical_cpus * config["cpuPercent"] // 100
        if logical_cpus > 64 or count < 1:
            return False
        affinity = (1 << count) - 1
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.OpenJobObjectW.restype = wintypes.HANDLE
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        job = kernel.OpenJobObjectW(4, False, name)  # JOB_OBJECT_QUERY
        if not job:
            return False
        try:
            process = kernel.GetCurrentProcess()
            member = wintypes.BOOL()
            if not kernel.IsProcessInJob(process, job, ctypes.byref(member)) or not member.value:
                return False
            rate = (wintypes.DWORD * 2)()
            if not kernel.QueryInformationJobObject(job, 15, ctypes.byref(rate), ctypes.sizeof(rate), None) or list(rate) != [0, 0]:
                return False

            class Limits(ctypes.Structure):
                _fields_ = [
                    ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
                    ("maximum_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD),
                ]

            limits = Limits()
            if not kernel.QueryInformationJobObject(job, 2, ctypes.byref(limits), ctypes.sizeof(limits), None):
                return False
            if limits.flags != 48 or limits.affinity != affinity or limits.priority != 16384:
                return False
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            return bool(
                kernel.GetProcessAffinityMask(process, ctypes.byref(process_mask), ctypes.byref(system_mask))
                and process_mask.value == affinity
            )
        finally:
            kernel.CloseHandle(job)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


@lru_cache(maxsize=1)
def _enforce_local_cpu_ceiling() -> None:
    """Source-checkout work cannot bypass the local Windows process-tree cap."""
    import os
    import subprocess

    if os.name != "nt":
        return
    checkout = Path(__file__).resolve().parents[2]
    if not (checkout / "pyproject.toml").is_file():
        return  # Installed/deployed runtimes have their own host resource limits.
    if _verified_inherited_cpu_ceiling(checkout):
        return
    subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-File",
            str(checkout / "scripts/run_local_capped.ps1"),
            "-AttachRootId", str(os.getpid()),
        ],
        check=True, capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=subprocess.STARTUPINFO(
            dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=0,
        ),
    )


def platform_root(explicit: str | Path | None = None) -> Path:
    """Return a runtime root containing configs and schemas.

    Source checkouts use the repository root. Installed wheels use generated
    package resources. An explicit path or BALOOT_ROOT always wins.
    """
    import os

    _enforce_local_cpu_ceiling()
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    configured = os.environ.get("BALOOT_ROOT", "").strip()
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parent / "resources",
        )
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "configs/bundle.yaml").is_file() and (
            resolved / "schemas/profile.schema.json"
        ).is_file():
            return resolved
    raise RuntimeError(
        "Baloot runtime profiles were not found; set BALOOT_ROOT or install package resources"
    )


def data_root(explicit: str | Path | None = None) -> Path:
    """Return a writable application-data directory.

    Container deployments normally set BALOOT_DATA_DIR, BALOOT_RUNTIME and
    BALOOT_OUTPUTS explicitly. Installed wheels otherwise use the platform's
    user data location instead of trying to write into package resources.
    """
    import os

    if explicit is not None:
        result = Path(explicit)
    elif os.environ.get("BALOOT_DATA_DIR", "").strip():
        result = Path(os.environ["BALOOT_DATA_DIR"])
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        result = Path(os.environ["LOCALAPPDATA"]) / "BalootPlatform"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        result = base / "baloot-platform"
    result = result.expanduser().resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


PROFILE_FAMILIES = (
    "rules",
    "conventions",
    "analysis",
    "video",
    "game",
    "product",
    "security",
    "runtime",
    "strategy",
    "ml",
    "storage",
    "agents",
    "learning",
    "challenge",
    "history",
    "audit",
    "difficulty",
    "database",
    "duplicate",
    "belief",
    "match_value",
    "challenge_contracts",
    "identity",
    "shared",
    "live",
    "adjudication",
    "hitl",
    "certification",
    "benchmark",
    "operations",
    "configuration",
    "performance",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def validate_certification_adjudication_identity(
    certification: Mapping[str, Any], repository_root: str | Path
) -> None:
    expected = certification.get("adjudication_authority_identity")
    if expected is None:
        return
    expected = dict(expected)
    required = {
        "id",
        "version",
        "authority_sha256",
        "authoring_catalog_sha256",
        "profile_sha256",
    }
    if set(expected) != required:
        raise ValueError("certification adjudication authority identity is not closed")
    from .adjudication.research import AdjudicationResearchAuthorityV1

    root = Path(repository_root).resolve()
    relative = str(certification.get("adjudication_profile") or "")
    path = (root / "configs" / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("certification adjudication profile is missing")
    document = _load_yaml(path)
    authority = AdjudicationResearchAuthorityV1.from_mapping(document)
    actual = {
        "id": authority.profile_id,
        "version": authority.version,
        "authority_sha256": authority.sha256,
        "authoring_catalog_sha256": authority.authoring_catalog()["sha256"],
        "profile_sha256": _sha(document),
    }
    if expected != actual:
        raise ValueError("certification adjudication authority identity is stale")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"profile {path} must be a mapping")
    return value


def _merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            result[key] = (
                _merge(result[key], value) if key in result else copy.deepcopy(value)
            )
        return result
    return copy.deepcopy(overlay)


@dataclass(frozen=True, slots=True)
class ProfileArtifact:
    family: str
    path: str
    data: dict[str, Any]
    sha256: str

    @property
    def id(self) -> str:
        return str(self.data.get("profile", {}).get("id", self.family))

    @property
    def version(self) -> str:
        return str(self.data.get("profile", {}).get("version", "0.0.0"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "path": self.path,
            "id": self.id,
            "version": self.version,
            "status": self.data.get("profile", {}).get("status"),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class ProfileBundle:
    rules: RuleBook
    components: Mapping[str, ProfileArtifact]
    mode: str
    bundle_profile: dict[str, Any]
    sha256: str

    @property
    def convention(self) -> dict[str, Any]:
        return self.components["conventions"].data

    @property
    def analysis(self) -> dict[str, Any]:
        return self.components["analysis"].data

    @property
    def video(self) -> dict[str, Any]:
        return self.components["video"].data

    def component(self, family: str) -> dict[str, Any]:
        return self.components[family].data

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "mode": self.mode,
            "profile": copy.deepcopy(self.bundle_profile.get("profile", {})),
            "components": {
                "rules": {
                    "id": self.rules.id,
                    "version": self.rules.version,
                    "sha256": self.rules.sha256,
                    "status": self.rules.data["profile"].get("status"),
                },
                **{
                    family: artifact.to_dict()
                    for family, artifact in self.components.items()
                    if family != "rules"
                },
            },
        }

    def configuration_snapshot(self) -> dict[str, Any]:
        """Return the complete immutable behavior snapshot for generated evidence.

        The normal bundle view intentionally stays compact. Generated games need
        the complete validated payloads as well, so future audits do not depend
        on whichever configuration revision happens to be active later.
        """
        core = {
            "schemaVersion": 1,
            "bundleHash": self.sha256,
            "mode": self.mode,
            "profile": copy.deepcopy(self.bundle_profile.get("profile", {})),
            "components": {
                "rules": {
                    "sha256": self.rules.sha256,
                    "payload": copy.deepcopy(self.rules.data),
                },
                **{
                    family: {
                        "sha256": artifact.sha256,
                        "payload": copy.deepcopy(artifact.data),
                    }
                    for family, artifact in sorted(self.components.items())
                    if family != "rules"
                },
            },
        }
        return {**core, "sha256": _sha(core)}


def resolved_bundle_identity(
    bundle_row: Mapping[str, Any], component_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "bundleProfile": {
            "id": str(bundle_row.get("bundleId") or "row_bundle"),
            "version": str(bundle_row.get("version") or "0.0.0"),
            "status": str(bundle_row.get("status") or "published"),
        },
        "mode": str(bundle_row.get("mode") or "full"),
        "components": dict(component_hashes),
    }


def profile_bundle_from_resolved(
    resolved: Mapping[str, Any], *, root: Path | None = None
) -> ProfileBundle:
    """Build the runtime bundle from one governed resolved row bundle.

    Row revision identifiers and timestamps are provenance and are excluded.
    The runtime hash binds the effective payload identities together with the
    bundle profile id, version, lifecycle status, and selected mode.
    """

    resolved_root = platform_root(root)
    bundle_row = dict(resolved.get("bundle") or {})
    profiles = dict(resolved.get("profiles") or {})
    profile_revisions = dict(resolved.get("profileRevisions") or {})
    missing = set(PROFILE_FAMILIES) - set(profiles)
    extra = set(profiles) - set(PROFILE_FAMILIES)
    if missing or extra:
        raise ValueError(
            "resolved configuration bundle components mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    artifacts: dict[str, ProfileArtifact] = {}
    rules: RuleBook | None = None
    component_hashes: dict[str, str] = {}
    for family in PROFILE_FAMILIES:
        payload = validate_profile_payload(
            family,
            profiles[family],
            root=resolved_root,
            source=f"<configuration:{family}>",
        )
        payload_hash = _sha(payload)
        component_hashes[family] = payload_hash
        revision = dict(profile_revisions.get(family) or {})
        source = f"configuration:{revision.get('revisionId', 'unversioned')}"
        artifacts[family] = ProfileArtifact(
            family=family,
            path=source,
            data=payload,
            sha256=payload_hash,
        )
        if family == "rules":
            rules = RuleBook(payload)

    if rules is None:  # Defensive; the closed family check above should catch it.
        raise ValueError("resolved configuration bundle is missing rules")

    validate_research_profile_bundle(
        {family: artifact.data for family, artifact in artifacts.items()}
    )
    _validate_research_agent_bindings(
        artifacts["agents"].data,
        artifacts["challenge"].data,
    )
    load_bound_rule_source_catalog(
        artifacts["certification"].data,
        rules,
        resolved_root,
    )
    validate_certification_rules_identity(
        artifacts["certification"].data,
        rules,
    )
    validate_certification_adjudication_identity(
        artifacts["certification"].data,
        resolved_root,
    )

    identity = resolved_bundle_identity(bundle_row, component_hashes)
    return ProfileBundle(
        rules=rules,
        components=artifacts,
        mode=identity["mode"],
        bundle_profile={"profile": identity["bundleProfile"]},
        sha256=_sha(identity),
    )


@lru_cache(maxsize=len(PROFILE_FAMILIES))
def _profile_schema_registry(schema_json: str) -> Registry | None:
    """Prepare closed schema references, never cache payload acceptance."""
    schema = json.loads(schema_json)
    if not isinstance(schema, dict):
        return None
    uri = schema.get("$id")
    if (schema.get("$schema") != jsonschema.Draft202012Validator.META_SCHEMA["$id"]
            or not isinstance(uri, str) or not uri.startswith(("https://", "http://"))
            or "#" in uri):
        return None
    pending = [schema]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            reference = value.get("$ref", "#")
            if (not isinstance(reference, str) or not reference.startswith("#")
                    or "$dynamicRef" in value or "$recursiveRef" in value
                    or (value is not schema and ("$id" in value or "$schema" in value))):
                return None
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return Registry().with_resource(uri, DRAFT202012.create_resource(schema)).crawl()


def _validate_profile(
    data: dict[str, Any], schema: dict[str, Any], path: Path, *, family: str | None = None
) -> None:
    # Referenced schemas must retain the source's first-error ordering.
    registry = _profile_schema_registry(json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    validator = (jsonschema.Draft202012Validator(schema) if registry is None else
                 jsonschema.Draft202012Validator(schema, registry=registry))
    try:
        validator.validate(data)
    except jsonschema.ValidationError as exc:
        if exc.context:
            exc = best_match(exc.context)
        if family == "agents":
            from .agents.profiles import agent_profile_schema_error

            raise agent_profile_schema_error(path, exc) from exc
        dotted = ".".join(str(part) for part in exc.absolute_path)
        raise ValueError(
            f"invalid profile {path}: {dotted or '<root>'}: {exc.message}"
        ) from exc


_HISTORICAL_PRODUCT_PROFILE_HASHES = frozenset({
    "38b2eca1eda1bc9177afd3b1eeeb1d285cdf16097f44d6e2d1ec28416a77a647",
    "ee96d97b262448db7fc5f40a1565274e726094f1d0d9eff61ba9fa3d74811796",
    "6f79495f00e3c1bfd3732b62ce5488c043a197cdf57aca5a183dc7a7260a1b0f",
    "3d361a20c7700d8ef8951016703017c0a73d9c43f0b4c9a3034da758afdf0ee3",
    "8d2f6aa5a1827b34ea39b61cfa0a0c0da0f4aa78f8fbcec3d6c999eb8d4223f3",
    "ab78d5232e3bf71f24c13747c64b369e7010ab39fd115dcc8973dc0b81ad1952",
})

_HISTORICAL_CONVENTIONS_PROFILE_HASHES = frozenset({
    # Frozen conservative_v1@1.0.0 row before the signal inventory expanded
    # from three to five governed signals. Only resolved immutable rows may use
    # this compatibility path; authored/current profiles remain schema-strict.
    "b28af1f969acb17a3e17adc3caf49db628758ebee90308dbe4edc3fab4fe276d",
})


def is_historical_conventions_profile(payload: Mapping[str, Any]) -> bool:
    """Identify an exact frozen conventions payload admitted for replay."""

    return _sha(payload) in _HISTORICAL_CONVENTIONS_PROFILE_HASHES


def _analysis_validation_ceiling() -> Any:
    """Return the process safety ceiling for validating one unbundled profile."""

    from baloot_platform.analysis.belief_v7 import (
        MAXIMUM_EXACT_ALLOCATION_LIMIT,
        MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
        BeliefPolicy,
    )

    return BeliefPolicy(
        profile_id="profile_validation_safety_ceiling",
        profile_version="1.0.0",
        requested_particles=1,
        maximum_particles=MAXIMUM_HIDDEN_DEAL_ATTEMPTS,
        exact_allocation_limit=1,
        maximum_exact_allocation_limit=MAXIMUM_EXACT_ALLOCATION_LIMIT,
        minimum_effective_fraction=1.0,
    )


def _validate_challenge_research_profiles(
    data: Mapping[str, Any],
    *,
    belief_policy: Any | None = None,
) -> None:
    """Use execution-owned typed builders for every Research profile block."""

    from baloot_platform.challenge.analysis_pipeline import (
        CascadedTraceAnalysisProfileV1,
        DecisionAnalysisExecutorSetV1,
        TraceAnalysisProfileV1,
        build_finalist_reanalysis_trace_profile_v1,
    )
    from baloot_platform.challenge.deep_analysis import DeepAnalysisProfileV1
    from baloot_platform.challenge.profiles import (
        ResearchComparisonProfileV1,
        SelectionProfileV1,
        ThrillProfileV1,
    )
    from baloot_platform.challenge.screening import OrganicScreeningProfileV1

    challenge = dict(data)
    confirmation = dict(challenge.get("confirmation") or {})
    screening = dict(challenge.get("screening") or {})
    stability = dict(challenge.get("stability") or {})
    bound_policy = belief_policy or _analysis_validation_ceiling()
    backend_id = "profile_publication_validation_v1"

    confirmation_profile = TraceAnalysisProfileV1.build(
        bid_samples=int(confirmation.get("bid_samples", 0)),
        play_samples=int(confirmation.get("deep_samples", 0)),
        exact_allocation_limit=int(confirmation.get("exact_allocation_limit", 0)),
        exact_search_max_remaining=int(confirmation.get("exact_search_remaining", -1)),
        maximum_sampling_attempts=int(confirmation.get("maximum_sampling_attempts", 0)),
        exact_search_max_nodes=int(confirmation.get("exact_search_max_nodes", 0)),
        exact_search_max_seconds=float(
            confirmation.get("exact_search_max_seconds", 0.0)
        ),
        exact_search_max_depth=int(confirmation.get("exact_search_max_depth", 0)),
        exact_search_max_cache_entries=int(
            confirmation.get("exact_search_max_cache_entries", -1)
        ),
        search_backend_id=backend_id,
        objective=str(confirmation.get("objective") or ""),
        use_conventions=False,
        belief_policy=bound_policy,
    )
    screening_profile = TraceAnalysisProfileV1.build(
        bid_samples=int(screening.get("shallow_samples", 0)),
        play_samples=int(screening.get("shallow_samples", 0)),
        exact_allocation_limit=int(screening.get("shallow_exact_allocation_limit", 0)),
        exact_search_max_remaining=int(
            screening.get("shallow_exact_search_remaining", -1)
        ),
        maximum_sampling_attempts=int(screening.get("maximum_sampling_attempts", 0)),
        exact_search_max_nodes=int(screening.get("exact_search_max_nodes", 0)),
        exact_search_max_seconds=float(screening.get("exact_search_max_seconds", 0.0)),
        exact_search_max_depth=int(confirmation.get("exact_search_max_depth", 0)),
        exact_search_max_cache_entries=int(
            confirmation.get("exact_search_max_cache_entries", -1)
        ),
        search_backend_id=backend_id,
        objective=str(confirmation.get("objective") or ""),
        use_conventions=False,
        belief_policy=bound_policy,
    )
    build_finalist_reanalysis_trace_profile_v1(
        stability,
        confirmation,
        search_backend_id=backend_id,
        belief_policy=bound_policy,
    )
    if confirmation.get("require_legal_replay") is not True:
        raise ValueError("challenge confirmation must require legal replay")
    deep = DeepAnalysisProfileV1.build(challenge.get("deep_analysis") or {})
    CascadedTraceAnalysisProfileV1.build(
        screening_profile,
        confirmation_profile,
        maximum_confirmation_decisions_per_trace=(
            deep.maximum_confirmation_decisions_per_trace
        ),
        minimum_screening_score=deep.minimum_screening_score,
    )
    DecisionAnalysisExecutorSetV1.build(
        challenge.get("decision_analysis_executor") or {}
    )
    SelectionProfileV1.build(challenge.get("selection_policy") or {})
    ThrillProfileV1.build(challenge.get("thrill_profile") or {})
    ResearchComparisonProfileV1.build(challenge.get("research_comparison") or {})
    OrganicScreeningProfileV1.build(challenge.get("campaign_screening") or {})


def validate_research_profile_bundle(profiles: Mapping[str, Any]) -> None:
    """Validate cross-profile Research compatibility before compilation/start."""

    from baloot_platform.analysis.belief_v7 import BeliefPolicy
    from baloot_platform.analysis.evaluator import AnalysisPolicy

    missing = {"analysis", "belief", "challenge"} - set(profiles)
    if missing:
        raise ValueError(
            f"Research profile bundle is missing families: {sorted(missing)}"
        )
    belief_policy = BeliefPolicy.from_profile(dict(profiles["belief"]))
    AnalysisPolicy.from_profile(dict(profiles["analysis"]), belief_policy=belief_policy)
    _validate_challenge_research_profiles(
        dict(profiles["challenge"]), belief_policy=belief_policy
    )


_SCENARIO_RESEARCH_V4_BUDGET_PROFILES = frozenset(
    {
        "interactive_validation_v1",
        "bounded_v1",
        "late_game_exact_v1",
        "actor_information_set_exact_v1",
        "offline_heavyweight_v1",
    }
)
_SCENARIO_RESEARCH_V4_PROOF_FIELDS = frozenset(
    {
        "maximumNodes",
        "maximumDepth",
        "maximumHorizonActions",
        "maximumHiddenWorlds",
        "maximumWallSeconds",
        "maximumCpuSeconds",
        "maximumCacheEntries",
        "maximumCacheBytes",
        "cancellationCheckIntervalNodes",
        "maximumWorkers",
        "maximumLineCountNodes",
        "maximumStateSpace",
    }
)
_SCENARIO_RESEARCH_V4_SEARCH_FIELDS = frozenset(
    {
        "maximumWallSeconds",
        "maximumDeterministicTime",
        "maximumConflicts",
        "maximumCandidates",
        "maximumPrefixLiftAttempts",
        "maximumVariants",
        "maximumMemoryBytes",
        "maximumWorkers",
        "maximumShardWorkers",
        "cancellationCheckIntervalMilliseconds",
    }
)


def _closed_mapping(
    value: Any, expected: set[str] | frozenset[str], name: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"challenge {name} shape is invalid")
    return dict(value)


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"challenge {name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError(f"challenge {name} must be positive")
    return float(value)


_NATIVE_RESEARCH_PROFILE_VERSIONS = (
    "3.0.0", "3.1.0", "3.2.0", "3.3.0", "3.4.0", "3.5.0", "3.6.0", "3.7.0", "3.8.0", "3.9.0", "3.10.0", "3.11.0", "3.12.0", "3.13.0", "3.14.0", "3.14.1", "3.14.2", "3.14.3", "3.14.4", "3.14.5", "3.14.6", "3.14.7",
)


def _validate_scenario_research_v4(
    data: Mapping[str, Any], profile_version: str
) -> None:
    present = "scenarioResearchV4" in data
    if profile_version not in ("2.9.0", *_NATIVE_RESEARCH_PROFILE_VERSIONS):
        if present:
            raise ValueError(
                f"challenge scenarioResearchV4 requires profile version 2.9.0 through {_NATIVE_RESEARCH_PROFILE_VERSIONS[-1]}"
            )
        return
    if not present:
        raise ValueError(
            f"challenge {profile_version} requires scenarioResearchV4"
        )

    block_fields = {
        "schemaVersion",
        "backend",
        "planner",
        "deterministicExecution",
        "policyCatalog",
        "proofBudgets",
        "searchBudgets",
        "diversityDefaults",
        "cacheLimits",
        "refusalThresholds",
    }
    if profile_version in _NATIVE_RESEARCH_PROFILE_VERSIONS[1:]:
        block_fields.add("authoringDefaults")
    if profile_version in _NATIVE_RESEARCH_PROFILE_VERSIONS[2:]:
        block_fields.add("qualityExecution")
    block = _closed_mapping(
        data["scenarioResearchV4"],
        block_fields,
        "scenarioResearchV4",
    )
    if block["schemaVersion"] != 1:
        raise ValueError("challenge scenarioResearchV4 schema version is unsupported")
    expected_backend = (
        "ortools_cp_sat_v2" if profile_version == "2.9.0" else "ortools_cp_sat_v3"
    )
    if block["backend"] != expected_backend:
        raise ValueError("challenge scenarioResearchV4 backend is unsupported")
    if "qualityExecution" in block:
        from baloot_platform.research.scenario_contracts import ScenarioQualityExecutionBudgetV1

        if expected_backend != "ortools_cp_sat_v3":
            raise ValueError("qualityExecution requires the native V3 backend")
        ScenarioQualityExecutionBudgetV1.build(block["qualityExecution"])
    if block["planner"] != "tail_first_v1":
        raise ValueError("challenge scenarioResearchV4 planner is unsupported")
    if block["deterministicExecution"] is not True:
        raise ValueError("challenge scenarioResearchV4 must be deterministic")

    quality_budget_profile = None
    quality_exact_profiles = {}
    if profile_version in _NATIVE_RESEARCH_PROFILE_VERSIONS[1:]:
        authoring = _closed_mapping(
            block["authoringDefaults"],
            {"schemaVersion", "resources", "playingQuality"},
            "scenarioResearchV4.authoringDefaults",
        )
        if authoring["schemaVersion"] != 1:
            raise ValueError(
                "challenge scenarioResearchV4 authoring defaults schema is unsupported"
            )
        authoring_resources = _closed_mapping(
            authoring["resources"],
            {"seedsPerShard"},
            "scenarioResearchV4.authoringDefaults.resources",
        )
        _positive_integer(
            authoring_resources["seedsPerShard"],
            "scenarioResearchV4.authoringDefaults.resources.seedsPerShard",
        )
        quality_fields = {"budgetProfile", "boundaryEvaluatorByDecision"}
        if "exactBudgetProfileByVisibility" in authoring["playingQuality"]:
            quality_fields.add("exactBudgetProfileByVisibility")
        quality_defaults = _closed_mapping(
            authoring["playingQuality"],
            quality_fields,
            "scenarioResearchV4.authoringDefaults.playingQuality",
        )
        quality_budget_profile = quality_defaults["budgetProfile"]
        if "exactBudgetProfileByVisibility" in quality_defaults:
            quality_exact_profiles = _closed_mapping(
                quality_defaults["exactBudgetProfileByVisibility"],
                {"complete_state", "actor_information_set"},
                "playingQuality.exactBudgetProfileByVisibility",
            )
        if type(quality_budget_profile) is not str or not quality_budget_profile:
            raise ValueError("challenge playing quality budget profile must be a string")
        quality_boundaries = authoring["playingQuality"]["boundaryEvaluatorByDecision"]
        quality_boundaries = _closed_mapping(
            quality_boundaries,
            {"bid", "escalation", "project", "play"},
            "scenarioResearchV4.authoringDefaults.playingQuality.boundaryEvaluatorByDecision",
        )
        from baloot_platform.research.scenario_contracts import PROOF_BOUNDARY_EVALUATOR_DEFINITIONS
        for decision_kind, evaluator in quality_boundaries.items():
            definition = PROOF_BOUNDARY_EVALUATOR_DEFINITIONS.get(evaluator)
            if definition is None or decision_kind not in definition["decisionKinds"]:
                raise ValueError("challenge playing quality boundary evaluator is unsupported")

    policies = _closed_mapping(
        block["policyCatalog"],
        {"schemaVersion", "defaultPolicyId", "profiles"},
        "scenarioResearchV4.policyCatalog",
    )
    if policies["schemaVersion"] != 1:
        raise ValueError("challenge scenarioResearchV4 policy catalog is unsupported")
    if policies["defaultPolicyId"] != "fixed_action_map_v1":
        raise ValueError("challenge scenarioResearchV4 default policy is unsupported")
    policy_profiles = _closed_mapping(
        policies["profiles"],
        {"fixed_action_map_v1"},
        "scenarioResearchV4.policyCatalog.profiles",
    )
    fixed_policy = _closed_mapping(
        policy_profiles["fixed_action_map_v1"],
        {"version", "materialization"},
        "scenarioResearchV4.policyCatalog.profiles.fixed_action_map_v1",
    )
    if (
        fixed_policy["version"] != "1.0.0"
        or fixed_policy["materialization"] != "seeded_legal_action_order_v1"
    ):
        raise ValueError("challenge scenarioResearchV4 policy identity is unsupported")

    proof = _closed_mapping(
        block["proofBudgets"],
        {"schemaVersion", "id", "version", "profiles"},
        "scenarioResearchV4.proofBudgets",
    )
    if (
        proof["schemaVersion"] != 1
        or proof["id"] != "configurable-scenario-proof-budgets"
        or proof["version"] not in {"2.0.0", "2.1.0"}
    ):
        raise ValueError("challenge scenarioResearchV4 proof budget identity is invalid")
    proof_profiles = _closed_mapping(
        proof["profiles"],
        _SCENARIO_RESEARCH_V4_BUDGET_PROFILES,
        "scenarioResearchV4.proofBudgets.profiles",
    )
    if quality_budget_profile is not None and quality_budget_profile not in proof_profiles:
        raise ValueError("challenge playing quality budget profile is unpublished")
    if any(type(value) is not str or value not in proof_profiles
           for value in quality_exact_profiles.values()):
        raise ValueError("challenge playing quality exact budget profile is unpublished")
    if (block.get("qualityExecution", {}).get("exactTailBudgetProfile") is not None
            and block["qualityExecution"]["exactTailBudgetProfile"] not in proof_profiles):
        raise ValueError("challenge exact tail budget profile is unpublished")
    proof_integer_fields = _SCENARIO_RESEARCH_V4_PROOF_FIELDS - {
        "maximumWallSeconds",
        "maximumCpuSeconds",
    }
    for profile_id, raw in proof_profiles.items():
        native = proof["version"] == "2.1.0"
        profile = _closed_mapping(
            raw,
            _SCENARIO_RESEARCH_V4_PROOF_FIELDS | ({"nativeObserved"} if native else set())
                | ({"policySearch"} if isinstance(raw, Mapping) and "policySearch" in raw
                   and native and expected_backend == "ortools_cp_sat_v3" else set()),
            f"scenarioResearchV4.proofBudgets.profiles.{profile_id}",
        )
        for field in proof_integer_fields:
            _positive_integer(profile[field], f"{profile_id}.{field}")
        _positive_number(profile["maximumWallSeconds"], f"{profile_id}.maximumWallSeconds")
        _positive_number(profile["maximumCpuSeconds"], f"{profile_id}.maximumCpuSeconds")
        if profile["maximumWorkers"] != 1:
            raise ValueError("challenge scenarioResearchV4 proof workers must equal one")
        if native:
            from .research.scenario_contracts import NativeObservedProofLimitsV1
            NativeObservedProofLimitsV1.build(profile["nativeObserved"])
        if "policySearch" in profile and profile["policySearch"] != "time_budgeted_symbolic_v1":
            raise ValueError("challenge proof policy search is unsupported")

    search = _closed_mapping(
        block["searchBudgets"],
        {"schemaVersion", "profiles"},
        "scenarioResearchV4.searchBudgets",
    )
    if search["schemaVersion"] != 1:
        raise ValueError("challenge scenarioResearchV4 search budget schema is unsupported")
    search_profiles = _closed_mapping(
        search["profiles"],
        _SCENARIO_RESEARCH_V4_BUDGET_PROFILES,
        "scenarioResearchV4.searchBudgets.profiles",
    )
    search_number_fields = {"maximumWallSeconds", "maximumDeterministicTime"}
    for profile_id, raw in search_profiles.items():
        profile = _closed_mapping(
            raw,
            _SCENARIO_RESEARCH_V4_SEARCH_FIELDS,
            f"scenarioResearchV4.searchBudgets.profiles.{profile_id}",
        )
        for field in _SCENARIO_RESEARCH_V4_SEARCH_FIELDS - search_number_fields:
            _positive_integer(profile[field], f"{profile_id}.{field}")
        for field in search_number_fields:
            _positive_number(profile[field], f"{profile_id}.{field}")
        if profile["maximumWorkers"] != 1:
            raise ValueError("challenge scenarioResearchV4 search workers must equal one")
        if profile["maximumShardWorkers"] > 32:
            raise ValueError(
                "challenge scenarioResearchV4 shard workers must not exceed 32"
            )
        if profile["maximumVariants"] > profile["maximumCandidates"]:
            raise ValueError(
                "challenge scenarioResearchV4 variants cannot exceed candidates"
            )

    diversity = _closed_mapping(
        block["diversityDefaults"],
        {
            "symmetryPolicy",
            "allowSymmetryOnlyVariants",
            "minimumDecisionStateHammingDistance",
            "minimumPrincipalVariationActionDistance",
        },
        "scenarioResearchV4.diversityDefaults",
    )
    if diversity["symmetryPolicy"] != "rules_preserving_v1":
        raise ValueError("challenge scenarioResearchV4 symmetry policy is unsupported")
    if diversity["allowSymmetryOnlyVariants"] is not False:
        raise ValueError("challenge scenarioResearchV4 symmetry-only variants are forbidden")
    for field in (
        "minimumDecisionStateHammingDistance",
        "minimumPrincipalVariationActionDistance",
    ):
        _positive_integer(diversity[field], f"diversityDefaults.{field}")

    cache = _closed_mapping(
        block["cacheLimits"],
        {"maximumEntries", "maximumBytes"},
        "scenarioResearchV4.cacheLimits",
    )
    for field, value in cache.items():
        _positive_integer(value, f"cacheLimits.{field}")

    refusal = _closed_mapping(
        block["refusalThresholds"],
        {
            "maximumModelVariables",
            "maximumModelConstraints",
            "maximumEstimatedStates",
            "maximumHiddenWorlds",
            "maximumLineCountLimbs",
        },
        "scenarioResearchV4.refusalThresholds",
    )
    for field, value in refusal.items():
        _positive_integer(value, f"refusalThresholds.{field}")


def _semantic_validate(family: str, data: dict[str, Any]) -> None:
    if family == "video":
        render = data.get("render", {})
        width = int(render.get("width", 0))
        height = int(render.get("height", 0))
        if width < 320 or height < 320:
            raise ValueError("video dimensions are too small")
        if width * 16 != height * 9:
            raise ValueError(
                "release video dimensions must use an exact 9:16 aspect ratio"
            )
        if render.get("locale") != "ar-SA" or render.get("direction") != "rtl":
            raise ValueError("release video must use Arabic ar-SA RTL presentation")
        if int(render.get("fps", 0)) not in {24, 25, 30, 50, 60}:
            raise ValueError("video fps must be one of 24,25,30,50,60")
        _validate_video_copy_templates(data)
        from baloot_platform.video.render_bundle import validate_majlis_profile

        validate_majlis_profile(data)
    elif family == "analysis":
        from baloot_platform.analysis.evaluator import AnalysisPolicy

        AnalysisPolicy.from_profile(data, belief_policy=_analysis_validation_ceiling())
    elif family == "security":
        if int(data.get("http", {}).get("maximum_body_bytes", 0)) <= 0:
            raise ValueError("security maximum_body_bytes must be positive")
    elif family == "runtime":
        if int(data.get("service", {}).get("port", 0)) not in range(1, 65536):
            raise ValueError("runtime service port is invalid")
        host = data.get("plugin_host", {})
        if not host.get("allowed_isolation") or not host.get("allowed_permissions"):
            raise ValueError(
                "runtime plugin host policy must declare isolation and permissions"
            )
        if (
            int(host.get("maximum_timeout_seconds", 0)) <= 0
            or int(host.get("maximum_memory_mb", 0)) < 64
        ):
            raise ValueError("runtime plugin resource policy is invalid")
    elif family == "learning":
        fractions = data.get("dataset", {})
        total = sum(
            float(fractions.get(name, 0.0))
            for name in ("train_fraction", "validation_fraction", "test_fraction")
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError("learning dataset split fractions must sum to 1")
    elif family == "database":
        jobs = data.get("jobs", {})
        lease = int(jobs.get("lease_seconds", 0))
        heartbeat = int(jobs.get("heartbeat_seconds", 0))
        attempts = int(jobs.get("maximum_attempts", 0))
        if lease < 1 or heartbeat < 1 or attempts < 1:
            raise ValueError(
                "database job lease, heartbeat, and retry policy must be positive"
            )
        if heartbeat * 3 > lease:
            raise ValueError(
                "database heartbeat must be no slower than one-third of the lease"
            )
    elif family == "challenge":
        profile = data.get("profile") or {}
        profile_identity = (profile.get("id"), profile.get("version"))
        legacy_versions = {
            "2.0.0",
            "2.1.0",
            "2.2.0",
            "2.3.0",
            "2.4.0",
            "2.5.0",
            "2.6.0",
            "2.7.0",
            "2.8.0",
            "2.8.1",
            "2.8.2",
            "2.8.3",
            "2.8.4",
            "2.9.0",
        }
        if profile_identity not in {
            *(("progressive_challenge_mining_v2", item) for item in legacy_versions),
            *(("progressive_challenge_mining_v3", item) for item in _NATIVE_RESEARCH_PROFILE_VERSIONS),
        }:
            raise ValueError("challenge profile identity/version is unsupported")
        profile_version = str(profile.get("version", "1"))
        _validate_scenario_research_v4(data, profile_version)
        publication = data.get("publication", {})
        if not 0 <= float(publication.get("minimum_publishability", 0.0)) <= 1:
            raise ValueError("challenge minimum_publishability must be between 0 and 1")
        if int(publication.get("reviewer_quorum", 0)) < 1:
            raise ValueError("challenge reviewer_quorum must be positive")
        for key in ("minimum_robustness", "minimum_archetype_score"):
            if not 0.0 <= float(publication.get(key, -1.0)) <= 1.0:
                raise ValueError(f"challenge {key} must be between 0 and 1")
        if (
            not publication.get("reviewer_roles")
            or not publication.get("publisher_roles")
        ):
            raise ValueError("challenge review and publisher roles must be declared")
        if (
            publication.get("require_governor_independence") is not True
        ):
            raise ValueError("challenge governor independence must be enabled")
        _validate_challenge_research_profiles(data)
        request_defaults = data.get("request_defaults")
        if request_defaults is not None:
            if request_defaults.get("source") not in {
                "simulation",
                "history",
                "both",
            }:
                raise ValueError("challenge default source is invalid")
            if request_defaults.get("decision_kind") not in {
                "play",
                "bid",
                "any",
            }:
                raise ValueError("challenge default decision kind is invalid")
            if int(request_defaults.get("seeds", 0)) < 1:
                raise ValueError("challenge default seed count is invalid")
            if (
                not 0
                <= int(request_defaults.get("difficulty_min", -1))
                <= int(request_defaults.get("difficulty_max", -1))
                <= 100
            ):
                raise ValueError("challenge default difficulty range is invalid")
            if (
                not 0.0
                <= float(
                    request_defaults.get(
                        "minimum_publishability",
                        -1.0,
                    )
                )
                <= 1.0
            ):
                raise ValueError("challenge default publishability is invalid")
            if int(request_defaults.get("maximum_results", 0)) < 1:
                raise ValueError("challenge default result limit is invalid")
            if float(request_defaults.get("maximum_seconds", 0.0)) <= 0.0:
                raise ValueError("challenge default deadline is invalid")
        stability = data.get("stability")
        if stability is not None:
            samples = int(stability.get("reanalysis_samples", 0))
            attempts = int(stability.get("maximum_sampling_attempts", 0))
            if samples < 2 or attempts < samples:
                raise ValueError("challenge stability sampling budget is invalid")
            if int(stability.get("exact_allocation_limit", 0)) < 1:
                raise ValueError("challenge stability allocation limit is invalid")
            if not 0 <= int(stability.get("exact_search_remaining", -1)) <= 32:
                raise ValueError(
                    "challenge stability exact-search threshold is invalid"
                )
            if int(stability.get("exact_search_max_nodes", 0)) < 1:
                raise ValueError("challenge stability node budget is invalid")
            if float(stability.get("exact_search_max_seconds", 0.0)) <= 0.0:
                raise ValueError("challenge stability safety timeout is invalid")
            if int(stability.get("seed_offset", 0)) < 1:
                raise ValueError("challenge stability seed offset is invalid")
            maximum_cohorts = int(stability.get("maximum_cohorts", 0))
            minimum_agreement = int(stability.get("minimum_agreement", 0))
            third_on_disagreement = stability.get("third_cohort_on_disagreement")
            if not 2 <= maximum_cohorts <= 3:
                raise ValueError("challenge stability cohort bound must be in [2, 3]")
            if not 2 <= minimum_agreement <= maximum_cohorts:
                raise ValueError("challenge stability agreement threshold is invalid")
            if not isinstance(third_on_disagreement, bool):
                raise ValueError(
                    "challenge stability third-cohort policy must be boolean"
                )
            if maximum_cohorts == 3 and not third_on_disagreement:
                raise ValueError(
                    "three-cohort stability requires adaptive third-cohort analysis"
                )
            if float(stability.get("minimum_regret_gap", -1.0)) < 0.0:
                raise ValueError("challenge stability regret gap is invalid")
            if not 0.0 <= float(stability.get("minimum_best_share", -1.0)) <= 1.0:
                raise ValueError("challenge stability best-share threshold is invalid")
            if (
                float(stability.get("maximum_competence_expected_margin_regret", -1.0))
                < 0.0
            ):
                raise ValueError("challenge competence regret threshold is invalid")
            if (
                not 0.0
                <= float(stability.get("maximum_competence_match_win_drop", -1.0))
                <= 1.0
            ):
                raise ValueError("challenge competence match-win threshold is invalid")
            if (
                float(
                    stability.get(
                        "maximum_expected_margin_drift",
                        -1.0,
                    )
                )
                < 0.0
            ):
                raise ValueError("challenge stability drift threshold is invalid")
            if int(stability.get("minimum_hidden_deals", 0)) < 2:
                raise ValueError("challenge stability hidden-deal support is invalid")
        screening = data.get("screening") or {}
        if int(screening.get("minimum_legal_actions", 0)) < 2:
            raise ValueError("challenge screening requires at least two legal actions")
        if not 0.0 < float(screening.get("retain_fraction", 0.0)) <= 1.0:
            raise ValueError("challenge screening retain_fraction must be in (0, 1]")
        diversity = data.get("diversity") or {}
        if int(diversity.get("fingerprint_version", 0)) < 1:
            raise ValueError("challenge fingerprint_version must be positive")
        if int(diversity.get("maximum_per_fingerprint", 0)) != 1:
            raise ValueError(
                "challenge currently supports exactly one item per fingerprint"
            )
        if int(diversity.get("maximum_per_archetype_per_batch", 0)) < 1:
            raise ValueError("challenge archetype batch limit must be positive")
        if not diversity.get("cell_dimensions"):
            raise ValueError("challenge quality-diversity cells cannot be empty")
        if int(diversity.get("top_k_per_cell", 0)) < 1:
            raise ValueError("challenge quality-diversity top-K must be positive")
        if int(diversity.get("rejected_audit_modulus", 0)) < 1:
            raise ValueError("challenge rejected-audit sample modulus must be positive")
        confirmation = data.get("confirmation") or {}
        if (
            not 0.0
            <= float(confirmation.get("minimum_confirmation_confidence", -1.0))
            <= 1.0
        ):
            raise ValueError("challenge confirmation confidence is out of range")
        guided = data.get("guided_search")
        if guided is not None:
            if not 1 <= int(guided.get("candidatesPerSeed", 0)) <= 1000:
                raise ValueError("guided search candidates per seed are out of range")
            if not 0.0 <= float(guided.get("minimumScore", -1.0)) <= 100.0:
                raise ValueError("guided search minimum score is out of range")
            if not guided.get("weights"):
                raise ValueError("guided search requires weighted structural targets")
            if str(guided.get("targetMode") or "") not in {"any", "all"}:
                raise ValueError("guided search target mode is invalid")
        disagreement = data.get("disagreement_search")
        if disagreement is not None:
            if not 1 <= int(disagreement.get("candidatesPerSeed", 0)) <= 100:
                raise ValueError(
                    "disagreement search candidates per seed are out of range"
                )
            if (
                not 1
                <= int(disagreement.get("maximumDeepDecisionsPerCandidate", 0))
                <= 64
            ):
                raise ValueError("disagreement deep-decision bound is out of range")
            if not 0.0 <= float(disagreement.get("minimumDeepConfidence", -1.0)) <= 1.0:
                raise ValueError("disagreement confidence gate is out of range")
            if not disagreement.get("weights"):
                raise ValueError("disagreement search requires weighted evidence")
        context_search = data.get("context_search")
        if context_search is not None:
            if (
                not 0.0
                <= float(context_search.get("minimumAnalysisConfidence", -1.0))
                <= 1.0
            ):
                raise ValueError("context search confidence gate is out of range")
            if (
                not 0.0
                <= float(
                    context_search.get("minimumMatchProbabilityAmplification", -1.0)
                )
                <= 1.0
            ):
                raise ValueError("context search amplification gate is out of range")
            if not 1 <= int(context_search.get("maximumContexts", 0)) <= 10_000:
                raise ValueError("context search bound is out of range")
        sakka_generation = data.get("sakka_generation")
        if sakka_generation is not None:
            if int(sakka_generation.get("target", 0)) < 1:
                raise ValueError("natural Sakka generation target must be positive")
            if not 1 <= int(sakka_generation.get("maximumRounds", 0)) <= 1_000:
                raise ValueError("natural Sakka generation round bound is out of range")
            if sakka_generation.get("seedPolicy") != "sha256_round_ordinal_v1":
                raise ValueError("unsupported natural Sakka generation seed policy")
        answer_policy = data.get("answer_policy")
        if answer_policy is not None:
            if not answer_policy.get("allowedTruthLabels"):
                raise ValueError("challenge answer policy requires truth labels")
            if not 0.0 <= float(answer_policy.get("minimumConfidence", -1.0)) <= 1.0:
                raise ValueError("challenge answer confidence gate is out of range")
        trace_import = data.get("trace_import_policy")
        if trace_import is not None:
            if not 1 <= int(trace_import.get("maximumEntries", 0)) <= 1_000_000:
                raise ValueError("trace import catalog bound is out of range")
            if not bool(trace_import.get("requireCompleteHiddenState", False)):
                raise ValueError("trace imports require complete hidden state")
        reachability = data.get("score_reachability")
        if reachability is not None:
            if not 1 <= int(reachability.get("maximumWitnessTraces", 0)) <= 1_000_000:
                raise ValueError("score reachability witness bound is out of range")
            if not 1 <= int(reachability.get("witnessesPerOutcome", 0)) <= 100:
                raise ValueError(
                    "score reachability witnesses per outcome is out of range"
                )
            if not 1 <= int(reachability.get("maximumRounds", 0)) <= 10_000:
                raise ValueError("score reachability round bound is out of range")
            if not 1 <= int(reachability.get("maximumStates", 0)) <= 10_000_000:
                raise ValueError("score reachability state bound is out of range")
        tactical = data.get("tactical_detectors")
        if tactical is not None:
            if "nativeDelayedPayoff" in tactical:
                from .tactics import TacticalDetectorProfileV1

                TacticalDetectorProfileV1.build(tactical)
                if (data.get("scenarioResearchV4") or {}).get("backend") != "ortools_cp_sat_v3":
                    raise ValueError("nativeDelayedPayoff requires the native V3 Research backend")
            low = float(tactical.get("lowCardMaxStrength", -1))
            high = float(tactical.get("highCardMinStrength", -1))
            if not 0 <= low < high <= 100:
                raise ValueError(
                    "tactical detector card-strength boundaries are invalid"
                )
            for key in (
                "minimumDelayedGain",
                "minimumCausalGain",
                "projectConsequenceDelta",
                "debateMarginGap",
            ):
                if not 0 <= float(tactical.get(key, -1)) <= 100:
                    raise ValueError(f"tactical detector {key} is out of range")
            for key in (
                "kaboutProbabilityDelta",
                "contractProbabilityDelta",
                "debateDisagreement",
            ):
                if not 0 <= float(tactical.get(key, -1)) <= 1:
                    raise ValueError(f"tactical detector {key} is out of range")
        perspective = data.get("perspective_policy")
        if perspective is not None:
            if perspective.get("actorPrompt") != "choose_acting_action":
                raise ValueError("acting-player truth requires its governed prompt")
            if not perspective.get("seatExperiencePrompts"):
                raise ValueError("seat-experience prompts cannot be empty")
            if not bool(perspective.get("requireFourProjections", False)):
                raise ValueError("all four observer projections are mandatory")
        campaign_proof = data.get("campaign_proof")
        if campaign_proof is not None:
            if (
                not 1
                <= int(campaign_proof.get("minimumOrganicRounds", 0))
                <= 1_000_000_000
            ):
                raise ValueError("campaign proof round target is out of range")
            workers = tuple(campaign_proof.get("requiredWorkerCounts") or ())
            schedules = tuple(campaign_proof.get("requiredFailureSchedules") or ())
            if not workers or any(not 1 <= int(item) <= 100_000 for item in workers):
                raise ValueError("campaign proof worker counts are invalid")
            if not schedules:
                raise ValueError("campaign proof failure schedules are required")
            if not bool(campaign_proof.get("requireAttemptLedger", False)):
                raise ValueError("campaign proof requires a worker-attempt ledger")
    elif family == "agents":
        profiles = data.get("profiles", {})
        if not profiles or data.get("default_profile") not in profiles:
            raise ValueError("agents profile requires a valid default_profile")
    elif family == "audit":
        allowed = set(data.get("modes", {}).get("allowed", []))
        default = data.get("modes", {}).get("default")
        if default not in allowed:
            raise ValueError("audit default mode must be included in allowed modes")
        if int(data.get("analysis", {}).get("requested_samples", 0)) <= 0:
            raise ValueError("audit requested_samples must be positive")
    elif family == "difficulty":
        model = data.get("model", {})
        if float(model.get("discrimination_min", 0.0)) <= 0:
            raise ValueError("difficulty discrimination_min must be positive")
        if float(model.get("discrimination_max", 0.0)) < float(
            model.get("discrimination_min", 0.0)
        ):
            raise ValueError("difficulty discrimination bounds are invalid")
    elif family == "database":
        if data.get("backend", {}).get("authoritative") != "postgresql":
            raise ValueError("database authoritative backend must be postgresql")
        if not bool(data.get("events", {}).get("append_only", False)):
            raise ValueError("database events must be append-only")
    elif family == "duplicate":
        modes = set(data.get("modes", []))
        if data.get("default_mode") not in modes:
            raise ValueError("duplicate default_mode must be enabled")
        if not bool(data.get("board", {}).get("immutable_after_publish", False)):
            raise ValueError("duplicate boards must be immutable after publication")
    elif family == "belief":
        from baloot_platform.analysis.belief_v7 import BeliefPolicy

        BeliefPolicy.from_profile(data)
    elif family == "match_value":
        if int(data.get("target", 0)) <= 0:
            raise ValueError("match value target must be positive")
        future_rounds = dict(data.get("future_rounds", {}))
        if future_rounds.get("solver") != "exact_dynamic_program_v2":
            raise ValueError("match value solver must be exact_dynamic_program_v2")
        if int(future_rounds.get("maximum_cache_states", 0)) <= 0:
            raise ValueError("match value cache limit must be positive")
    elif family == "identity":
        from baloot_platform.identity.pagination import IdentityPaginationConfig

        if data.get("mode") not in {"optional", "required", "disabled"}:
            raise ValueError("identity mode must be optional, required or disabled")
        if int(data.get("local_accounts", {}).get("password_minimum", 0)) < 8:
            raise ValueError("identity password minimum must be at least 8")
        IdentityPaginationConfig.model_validate(data["pagination"])
    elif family == "shared":
        schedule = data.get("schedule", {})
        if int(schedule.get("minimum_deal_attempts", 0)) <= 0:
            raise ValueError("shared minimum_deal_attempts must be positive")
        if int(schedule.get("maximum_deal_attempts", 0)) < int(
            schedule.get("minimum_deal_attempts", 0)
        ):
            raise ValueError(
                "shared maximum_deal_attempts must cover minimum_deal_attempts"
            )
        allowed = set(data.get("sharing", {}).get("allowed_scopes", []))
        if data.get("sharing", {}).get("default_scope") not in allowed:
            raise ValueError("shared default_scope must be allowed")
        if data.get("terminology") != {
            "match_ar": "صكّة",
            "round_ar": "لفة",
            "trick_ar": "أكلة",
        }:
            raise ValueError("shared Arabic terminology must use صكّة/لفة/أكلة")
    elif family == "challenge_contracts":
        contracts = data.get("contracts", {})
        required = {
            "outcome_flip",
            "actual_deal_flip",
            "big_win_small_win",
            "comeback",
            "debate",
            "damage_control",
            "buy_or_pass",
        }
        missing = required - set(contracts)
        if missing:
            raise ValueError(f"challenge contract profile is missing {sorted(missing)}")
        for contract_name, settings in contracts.items():
            if not isinstance(settings, dict):
                raise ValueError(
                    f"challenge contract {contract_name} must be a mapping"
                )
            for key, value in settings.items():
                if (
                    key.endswith("_probability_min")
                    or key.endswith("_probability_max")
                    or key
                    in {
                        "probability_swing_min",
                        "maximum_utility_gap",
                        "minimum_hidden_deal_disagreement",
                    }
                ):
                    if not 0.0 <= float(value) <= 1.0:
                        raise ValueError(
                            f"challenge contract {contract_name}.{key} must be between 0 and 1"
                        )
        outcome = contracts["outcome_flip"]
        if float(outcome["alternative_match_win_probability_max"]) >= float(
            outcome["best_match_win_probability_min"]
        ):
            raise ValueError("challenge outcome_flip probability boundaries overlap")
        publication = data.get("publication") or {}
        if not all(
            publication.get(key) is True
            for key in (
                "require_verified_contract",
                "prohibit_untyped_challenges",
                "require_confidence_intervals",
                "require_provenance",
            )
        ):
            raise ValueError(
                "challenge truth publication gates must remain fail-closed"
            )
    elif family == "hitl":
        review = data.get("review", {})
        if int(review.get("minimum_independent_reviewers", 0)) < 2:
            raise ValueError("hitl minimum reviewers must be at least two")
        if not 0.0 <= float(review.get("minimum_weighted_support", -1)) <= 1.0:
            raise ValueError("hitl weighted support must be between zero and one")
        if bool(data.get("training", {}).get("automatic_model_promotion", True)):
            raise ValueError("hitl cannot automatically promote production models")
        if bool(data.get("improvement", {}).get("automatic_production_apply", True)):
            raise ValueError("hitl cannot automatically mutate production")
    elif family == "certification":
        if bool(data.get("automatic_activation", True)):
            raise ValueError(
                "certification candidates cannot activate rules automatically"
            )
        if int(data.get("minimum_external_signatures", 0)) < 2:
            raise ValueError("certification requires independent external signatures")
        certification_eligible = data.get("certification_eligible", True)
        if not isinstance(certification_eligible, bool):
            raise ValueError("certification eligibility must be a boolean")
        trusted_signers = list(data.get("trusted_signers") or ())
        signer_ids = [str(item.get("signer_id", "")) for item in trusted_signers]
        if len(signer_ids) != len(set(signer_ids)) or any(not item for item in signer_ids):
            raise ValueError("certification trusted signer IDs must be non-empty and unique")
        required_roles = set(data.get("required_signer_roles") or ())
        if any(str(item.get("role", "")) not in required_roles for item in trusted_signers):
            raise ValueError("certification trusted signer roles must be required roles")
    elif family == "benchmark":
        if bool(data.get("automatic_promotion", True)):
            raise ValueError("benchmark results cannot automatically promote models")
        if int(data.get("promotion_thresholds", {}).get("minimum_examples", 0)) <= 0:
            raise ValueError("benchmark minimum examples must be positive")
        paired = data.get("paired_comparison", {})
        if int(paired.get("minimum_pairs", 0)) <= 0:
            raise ValueError("benchmark minimum pairs must be positive")
        if float(paired.get("maximum_seconds", 0.0)) <= 0:
            raise ValueError("benchmark maximum seconds must be positive")
        candidate_profile = str(paired.get("candidate_profile_id", "")).strip()
        baseline_profile = str(paired.get("baseline_profile_id", "")).strip()
        if (
            not candidate_profile
            or not baseline_profile
            or candidate_profile == baseline_profile
        ):
            raise ValueError("benchmark paired policies must be distinct and explicit")
        symmetry = data.get("symmetry", {})
        required_transforms = set(symmetry.get("required_transforms") or ())
        if required_transforms != {"seat_rotation", "team_swap", "suit_relabel"}:
            raise ValueError("benchmark symmetry transforms are incomplete")
        if int(symmetry.get("maximum_cases", 0)) <= 0:
            raise ValueError("benchmark symmetry cases must be positive")
        if float(symmetry.get("tolerance", -1.0)) < 0.0:
            raise ValueError("benchmark symmetry tolerance cannot be negative")
        local_preanalysis = data.get("local_preanalysis", {})
        if int(local_preanalysis.get("minimum_positions", 0)) <= 0:
            raise ValueError("local preanalysis minimum positions must be positive")
        if (
            not 0.0
            <= float(local_preanalysis.get("minimum_policy_accuracy", -1.0))
            <= 1.0
        ):
            raise ValueError(
                "local preanalysis minimum accuracy must be between zero and one"
            )
        if float(local_preanalysis.get("maximum_expected_regret", -1.0)) < 0.0:
            raise ValueError("local preanalysis regret limit cannot be negative")
        if float(local_preanalysis.get("maximum_illegal_action_rate", -1.0)) != 0.0:
            raise ValueError("local preanalysis must reject every illegal action")
        if float(local_preanalysis.get("maximum_p95_latency_ms", 0.0)) <= 0.0:
            raise ValueError("local preanalysis latency budget must be positive")
        if bool(local_preanalysis.get("qualification_eligible", True)):
            raise ValueError("local preanalysis cannot qualify a release")
    elif family == "operations":
        if not data.get("objectives"):
            raise ValueError("operations profile requires objectives")
        if bool(data.get("automatic_production_change", True)):
            raise ValueError(
                "operations evaluation cannot change production automatically"
            )
    elif family == "configuration":
        if data.get("storage", {}).get("authoritative") != "postgresql":
            raise ValueError("configuration authoritative storage must be postgresql")
        if not bool(data.get("storage", {}).get("append_only_versions", False)):
            raise ValueError("configuration versions must be append-only")
        if bool(data.get("governance", {}).get("automatic_production_apply", True)):
            raise ValueError("configuration cannot automatically mutate production")
    elif family == "performance":
        budgets = data.get("budgets", {})
        if not budgets or any(float(value) <= 0 for value in budgets.values()):
            raise ValueError("performance budgets must be positive")
        if int(data.get("measurement", {}).get("measured_iterations", 0)) < 10:
            raise ValueError("performance measurement requires at least ten iterations")


def load_profile_payload(
    family: str, path: str | Path, *, root: Path | None = None
) -> dict[str, Any]:
    """Load and validate a profile payload through the canonical schema boundary."""
    resolved_root = platform_root(root)
    source = Path(path)
    if not source.is_absolute():
        source = resolved_root / source
    data = _load_yaml(source)
    validate_profile_payload(family, data, root=resolved_root, source=source)
    return data


def validate_profile_payload(
    family: str,
    payload: Mapping[str, Any],
    *,
    root: Path | None = None,
    source: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an in-memory profile using the same closed schemas as file bundles."""
    resolved_root = platform_root(root)
    family = str(family)
    if family not in PROFILE_FAMILIES:
        raise ValueError(f"unknown profile family: {family}")
    if not isinstance(payload, Mapping):
        raise ValueError("profile payload must be a mapping")
    data = copy.deepcopy(dict(payload))
    if family == "rules":
        schema = json.loads(
            (resolved_root / "schemas/profiles/rules.schema.json").read_text(
                encoding="utf-8"
            )
        )
        _validate_profile(data, schema, Path(source or "<row:rules>"))
        RuleBook(data)
        return data
    if (
        family == "product"
        and str(source) == "<configuration:product>"
        and _sha(data) in _HISTORICAL_PRODUCT_PROFILE_HASHES
    ):
        return data
    if (
        family == "conventions"
        and str(source) == "<configuration:conventions>"
        and is_historical_conventions_profile(data)
    ):
        return data
    candidate = resolved_root / "schemas/profiles" / f"{family}.schema.json"
    schema_path = (
        candidate
        if candidate.exists()
        else resolved_root / "schemas/profile.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    _validate_profile(data, schema, Path(source or f"<row:{family}>"), family=family)
    _semantic_validate(family, data)
    return data


def _validate_research_agent_bindings(
    agents: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> None:
    """Fail closed when a Research lane can use a weak or stochastic lineup.

    Research construction and qualification intentionally use one profile for
    every seat.  This gate protects that profile boundary without changing the
    beginner, regular, or default online lineups.
    """
    profiles = dict(agents.get("profiles") or {})
    bindings = {
        "organic Research": dict(challenge.get("campaign_screening") or {}).get(
            "agentProfileId"
        ),
        "guided Research": dict(challenge.get("guided_search") or {}).get(
            "agentProfileId"
        ),
        "candidate resimulation": dict(
            challenge.get("candidate_resimulation") or {}
        ).get("agentProfileId"),
        "finalist replay": dict(challenge.get("finalist_validation") or {}).get(
            "agentProfileId"
        ),
    }
    required_positive = (
        "play_search_budget",
        "play_search_samples",
        "play_search_max_remaining",
        "play_exact_allocation_limit",
        "play_exact_search_remaining",
        "play_maximum_sampling_attempts",
        "bid_samples",
    )
    for stage, raw_profile_id in bindings.items():
        profile_id = str(raw_profile_id or "").strip()
        profile = dict(profiles.get(profile_id) or {})
        if not profile:
            raise ValueError(f"{stage} requires a declared governed agent profile")
        if float(profile.get("skill", -1.0)) != 1.0:
            raise ValueError(f"{stage} agent skill must be 1.0")
        if float(profile.get("mistake_probability", -1.0)) != 0.0:
            raise ValueError(f"{stage} agent must disable stochastic mistakes")
        if float(profile.get("deception_probability", -1.0)) != 0.0:
            raise ValueError(f"{stage} agent must disable stochastic deception")
        if float(profile.get("signal_discipline", -1.0)) != 1.0:
            raise ValueError(f"{stage} agent must use full signal discipline")
        if str(profile.get("play_search_mode") or "") != "paired_information_set":
            raise ValueError(
                f"{stage} agent must use observer-safe paired information-set search"
            )
        if any(int(profile.get(key, 0)) <= 0 for key in required_positive):
            raise ValueError(
                f"{stage} agent must declare positive search and bidding budgets"
            )
        if bool(profile.get("use_production_model", False)):
            raise ValueError(f"{stage} agent cannot use an unfrozen production model")
        if float(profile.get("model_weight", -1.0)) != 0.0:
            raise ValueError(f"{stage} agent model weight must be zero")

    discovery_ids = {
        str(bindings[stage])
        for stage in (
            "organic Research",
            "guided Research",
            "candidate resimulation",
        )
    }
    if len(discovery_ids) != 1:
        raise ValueError(
            "Research discovery lanes must use one identical governed agent profile"
        )


def build_bundle(
    root: Path | None = None,
    *,
    mode: str = "full",
    bundle_path: str | Path | None = None,
) -> ProfileBundle:
    root = platform_root(root)
    bundle_file = Path(bundle_path) if bundle_path else root / "configs/bundle.yaml"
    if not bundle_file.is_absolute():
        bundle_file = root / bundle_file
    bundle_data = _load_yaml(bundle_file)
    component_paths = bundle_data.get("components")
    if not isinstance(component_paths, dict):
        raise ValueError("bundle components must be a mapping")
    missing = set(PROFILE_FAMILIES) - set(component_paths)
    if missing:
        raise ValueError(f"bundle is missing component families: {sorted(missing)}")

    generic_schema = json.loads(
        (root / "schemas/profile.schema.json").read_text(encoding="utf-8")
    )
    artifacts: dict[str, ProfileArtifact] = {}
    rules_path = root / "configs" / str(component_paths["rules"])
    overlays: list[Path] = []
    if mode == "sun_only":
        overlays.append(root / "configs/modes/sun_only.yaml")
    elif mode == "hokum_only":
        overlays.append(root / "configs/modes/hokum_only.yaml")
    elif mode != "full":
        raise ValueError(f"unknown mode profile {mode}")
    rules = RuleBook.load(rules_path, *overlays)
    rules_schema_path = root / "schemas/profiles/rules.schema.json"
    rules_schema = json.loads(rules_schema_path.read_text(encoding="utf-8"))
    _validate_profile(rules.data, rules_schema, rules_path)
    artifacts["rules"] = ProfileArtifact(
        "rules", rules_path.relative_to(root).as_posix(), rules.data, rules.sha256
    )

    for family in PROFILE_FAMILIES:
        if family == "rules":
            continue
        path = root / "configs" / str(component_paths[family])
        data = _load_yaml(path)
        family_schema_path = root / "schemas" / "profiles" / f"{family}.schema.json"
        schema = (
            json.loads(family_schema_path.read_text(encoding="utf-8"))
            if family_schema_path.exists()
            else generic_schema
        )
        _validate_profile(data, schema, path, family=family)
        _semantic_validate(family, data)
        artifacts[family] = ProfileArtifact(
            family, path.relative_to(root).as_posix(), data, _sha(data)
        )

    validate_research_profile_bundle(
        {family: artifact.data for family, artifact in artifacts.items()}
    )
    _validate_research_agent_bindings(
        artifacts["agents"].data,
        artifacts["challenge"].data,
    )
    load_bound_rule_source_catalog(
        artifacts["certification"].data,
        rules,
        root,
    )
    validate_certification_rules_identity(
        artifacts["certification"].data,
        rules,
    )
    validate_certification_adjudication_identity(
        artifacts["certification"].data,
        root,
    )

    identity = {
        "bundleProfile": bundle_data.get("profile", {}),
        "mode": mode,
        "components": {family: artifacts[family].sha256 for family in PROFILE_FAMILIES},
    }
    return ProfileBundle(rules, artifacts, mode, bundle_data, _sha(identity))


def write_profile_manifest(
    root: Path | None = None, output: Path | None = None
) -> Path:
    root = platform_root(root)
    output = output or root / "generated/profile-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    bundles = {
        mode: build_bundle(root, mode=mode).to_dict()
        for mode in ("full", "sun_only", "hokum_only")
    }
    output.write_text(
        json.dumps(
            {"schemaVersion": 1, "bundles": bundles},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output

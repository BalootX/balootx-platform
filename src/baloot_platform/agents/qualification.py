"""Evidence-bounded Foundry agent release qualification."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .profiles import AgentProfile, AgentProfileRegistry


_GENERATED_SOURCE_DIRECTORIES = frozenset(
    {".codex-tmp", ".pytest_cache", "__pycache__", "node_modules"}
)


def workspace_source_paths(
    roots: tuple[Path, ...], suffixes: set[str] | frozenset[str]
) -> set[Path]:
    paths: set[Path] = set()
    for owner in roots:
        for directory, names, filenames in os.walk(owner):
            names[:] = [
                name for name in names if name not in _GENERATED_SOURCE_DIRECTORIES
            ]
            paths.update(
                Path(directory) / filename
                for filename in filenames
                if Path(filename).suffix.lower() in suffixes
            )
    return paths


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(value: Any, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


def _result_hash(value: Mapping[str, Any], field: str) -> str:
    expected = _sha256(value.get("resultHash"), field)
    payload = {key: item for key, item in value.items() if key != "resultHash"}
    if _canonical_hash(payload) != expected:
        raise ValueError(f"{field} checksum mismatch")
    return expected


def _document_hash(value: Mapping[str, Any], hash_field: str, field: str) -> str:
    expected = _sha256(value.get(hash_field), field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    if _canonical_hash(payload) != expected:
        raise ValueError(f"{field} checksum mismatch")
    return expected


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_finalist_analysis_compatibility(
    profile: AgentProfile,
    confirmation: Mapping[str, Any],
) -> None:
    """Reject a finalist policy that its own authority will predictably fail."""
    failures: list[str] = []
    if profile.play_search_mode != "paired_information_set":
        failures.append("play_search_mode")
    if profile.play_search_objective != str(confirmation.get("objective") or ""):
        failures.append("play_search_objective")
    if profile.play_search_max_remaining < 32:
        failures.append("play_search_max_remaining")
    if profile.play_search_samples < int(confirmation.get("deep_samples", 0)):
        failures.append("play_search_samples")
    if profile.play_exact_allocation_limit < int(
        confirmation.get("exact_allocation_limit", 0)
    ):
        failures.append("play_exact_allocation_limit")
    if profile.play_exact_search_remaining < int(
        confirmation.get("exact_search_remaining", 0)
    ):
        failures.append("play_exact_search_remaining")
    if profile.play_maximum_sampling_attempts < int(
        confirmation.get("maximum_sampling_attempts", 0)
    ):
        failures.append("play_maximum_sampling_attempts")
    if profile.bid_samples < int(confirmation.get("bid_samples", 0)):
        failures.append("bid_samples")
    if profile.mistake_probability != 0.0:
        failures.append("mistake_probability")
    if profile.deception_probability != 0.0:
        failures.append("deception_probability")
    if failures:
        raise ValueError(
            "finalist agent is incompatible with confirmation authority: "
            + ", ".join(failures)
        )


@dataclass(frozen=True, slots=True)
class FinalistAgentPilotEvidenceV1:
    """Content-bound local pilot evidence that can never qualify a release."""

    status: str
    passed: bool
    candidate_policy: str
    source_binding: str | None
    positions: int
    policy_accuracy: float | None
    expected_regret: float | None
    illegal_action_rate: float | None
    latency_p95_ms: float | None
    baseline_policy: str | None
    baseline_accuracy: float | None
    baseline_expected_regret: float | None
    evidence_hash: str | None
    source_manifest_hash: str | None
    qualification_eligible: bool
    issues: tuple[str, ...]
    sha256: str

    @classmethod
    def load(
        cls,
        *,
        root: Path,
        profile: AgentProfile,
        benchmark_profile: Mapping[str, Any],
        evidence_path: Path,
        source_manifest_path: Path,
    ) -> "FinalistAgentPilotEvidenceV1":
        if not evidence_path.is_file() or not source_manifest_path.is_file():
            missing = []
            if not evidence_path.is_file():
                missing.append("pilot_evidence_missing")
            if not source_manifest_path.is_file():
                missing.append("source_manifest_missing")
            return cls._summary(
                status="missing",
                candidate_policy=profile.id,
                issues=missing,
            )
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            return cls._validate(
                root=root.resolve(),
                profile=profile,
                benchmark_profile=benchmark_profile,
                evidence=evidence,
                manifest=manifest,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            message = str(exc).strip() or type(exc).__name__
            status = "stale" if message.startswith("stale:") else "invalid"
            return cls._summary(
                status=status,
                candidate_policy=profile.id,
                issues=(message,),
            )

    @classmethod
    def _validate(
        cls,
        *,
        root: Path,
        profile: AgentProfile,
        benchmark_profile: Mapping[str, Any],
        evidence: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> "FinalistAgentPilotEvidenceV1":
        evidence_value = dict(evidence)
        manifest_value = dict(manifest)
        evidence_hash = _result_hash(evidence_value, "pilotEvidence.resultHash")
        manifest_hash = _document_hash(
            manifest_value,
            "sha256",
            "pilotSourceManifest.sha256",
        )
        if int(evidence_value.get("schemaVersion", 0)) != 2:
            raise ValueError("pilot evidence schema is unsupported")
        if evidence_value.get("kind") != "observer_safe_exact_subset_preanalysis":
            raise ValueError("pilot evidence kind is unsupported")
        if evidence_value.get("status") != "passed":
            raise ValueError("pilot evidence did not pass")
        if int(manifest_value.get("schemaVersion", 0)) != 2:
            raise ValueError("pilot source manifest schema is unsupported")
        if manifest_value.get("kind") != "local_finalist_preanalysis_source_manifest":
            raise ValueError("pilot source manifest kind is unsupported")

        source = dict(evidence_value.get("source") or {})
        local_pilot = dict(evidence_value.get("localPilot") or {})
        if source.get("binding") != "local_workspace_manifest":
            raise ValueError("pilot evidence is not locally content-bound")
        if local_pilot.get("qualificationEligible") is not False:
            raise ValueError("pilot evidence must not be qualification eligible")
        if evidence_value.get("automaticPromotion") is not False:
            raise ValueError("pilot evidence must disable automatic promotion")
        if dict(evidence_value.get("promotionDecision") or {}).get("decision") != "not_eligible":
            raise ValueError("pilot evidence promotion boundary is invalid")
        if (
            source.get("workspaceManifestSha256") != manifest_hash
            or local_pilot.get("sourceManifestSha256") != manifest_hash
        ):
            raise ValueError("pilot source manifest binding mismatch")

        candidate_policy = str(local_pilot.get("candidatePolicy") or "")
        if candidate_policy != profile.id:
            raise ValueError("pilot candidate policy does not match the active profile")
        policies = dict(evidence_value.get("policies") or {})
        candidate = dict(policies.get(candidate_policy) or {})
        baseline_policy = str(local_pilot.get("baselinePolicy") or "") or None
        baseline = dict(policies.get(baseline_policy) or {}) if baseline_policy else {}
        if candidate.get("policyId") != candidate_policy:
            raise ValueError("pilot candidate aggregate is missing")
        if candidate.get("policySnapshotSha256") != _canonical_hash(profile.to_dict()):
            raise ValueError("stale: active finalist profile changed after the pilot")
        forbidden = {"decisions", "predictions", "positionResults", "rows"}
        if forbidden.intersection(evidence_value) or forbidden.intersection(candidate):
            raise ValueError("pilot evidence must contain aggregates only")

        cls._validate_manifest_freshness(root, manifest_value)
        policy = dict(benchmark_profile.get("local_preanalysis") or {})
        if not policy or policy.get("qualification_eligible") is not False:
            raise ValueError("local preanalysis policy must be non-qualifying")
        positions = int(candidate.get("positions", 0))
        accuracy = float(candidate.get("top1Accuracy", 0.0))
        regret = float(candidate.get("expectedRegret", float("inf")))
        illegal = float(candidate.get("illegalActionRate", 1.0))
        latency = float(dict(candidate.get("latencyMs") or {}).get("p95", float("inf")))
        checks = {
            "minimum_positions": positions >= int(policy.get("minimum_positions", 1)),
            "minimum_policy_accuracy": accuracy
            >= float(policy.get("minimum_policy_accuracy", 1.0)),
            "maximum_expected_regret": regret
            <= float(policy.get("maximum_expected_regret", 0.0)),
            "maximum_illegal_action_rate": illegal
            <= float(policy.get("maximum_illegal_action_rate", 0.0)),
            "maximum_p95_latency_ms": latency
            <= float(policy.get("maximum_p95_latency_ms", 0.0)),
            "all_ranked_actions_legal": candidate.get("allRankedActionsLegal") is True,
        }
        issues = tuple(sorted(f"pilot:{name}" for name, passed in checks.items() if not passed))
        passed = not issues
        return cls._summary(
            status="passed" if passed else "threshold_failed",
            passed=passed,
            candidate_policy=candidate_policy,
            source_binding="local_workspace_manifest",
            positions=positions,
            policy_accuracy=accuracy,
            expected_regret=regret,
            illegal_action_rate=illegal,
            latency_p95_ms=latency,
            baseline_policy=baseline_policy,
            baseline_accuracy=float(baseline.get("top1Accuracy", 0.0)) if baseline else None,
            baseline_expected_regret=float(baseline.get("expectedRegret", 0.0)) if baseline else None,
            evidence_hash=evidence_hash,
            source_manifest_hash=manifest_hash,
            issues=issues,
        )

    @staticmethod
    def _validate_manifest_freshness(root: Path, manifest: Mapping[str, Any]) -> None:
        coverage = dict(manifest.get("coverage") or {})
        suffixes = {str(item).lower() for item in coverage.get("suffixes") or ()}
        expected_paths: set[str] = set()
        for owner_value in coverage.get("roots") or ():
            owner = (root / str(owner_value)).resolve()
            if root != owner and root not in owner.parents:
                raise ValueError("pilot manifest coverage escapes the workspace")
            if owner.is_dir():
                expected_paths.update(
                    path.relative_to(root).as_posix()
                    for path in workspace_source_paths((owner,), suffixes)
                )
        expected_paths.update(str(item) for item in coverage.get("explicitPaths") or ())
        expected_paths.update(str(item) for item in coverage.get("inputPaths") or ())
        entries = {str(item.get("path")): dict(item) for item in manifest.get("files") or ()}
        if expected_paths != set(entries):
            raise ValueError("stale: pilot source file set changed")
        for relative, entry in entries.items():
            path = (root / relative).resolve()
            if root != path and root not in path.parents:
                raise ValueError("pilot manifest file escapes the workspace")
            if not path.is_file():
                raise ValueError(f"stale: pilot source file is missing: {relative}")
            if int(entry.get("bytes", -1)) != path.stat().st_size:
                raise ValueError(f"stale: pilot source file size changed: {relative}")
            if _sha256(entry.get("sha256"), f"pilotSourceFile[{relative}].sha256") != _file_hash(path):
                raise ValueError(f"stale: pilot source file changed: {relative}")

    @classmethod
    def _summary(
        cls,
        *,
        status: str,
        candidate_policy: str,
        passed: bool = False,
        source_binding: str | None = None,
        positions: int = 0,
        policy_accuracy: float | None = None,
        expected_regret: float | None = None,
        illegal_action_rate: float | None = None,
        latency_p95_ms: float | None = None,
        baseline_policy: str | None = None,
        baseline_accuracy: float | None = None,
        baseline_expected_regret: float | None = None,
        evidence_hash: str | None = None,
        source_manifest_hash: str | None = None,
        issues: tuple[str, ...] | list[str] = (),
    ) -> "FinalistAgentPilotEvidenceV1":
        issue_values = tuple(sorted(set(str(item) for item in issues)))
        core = {
            "schemaVersion": 1,
            "status": status,
            "passed": bool(passed),
            "candidatePolicy": candidate_policy,
            "sourceBinding": source_binding,
            "positions": positions,
            "policyAccuracy": policy_accuracy,
            "expectedRegret": expected_regret,
            "illegalActionRate": illegal_action_rate,
            "latencyP95Ms": latency_p95_ms,
            "baselinePolicy": baseline_policy,
            "baselineAccuracy": baseline_accuracy,
            "baselineExpectedRegret": baseline_expected_regret,
            "evidenceHash": evidence_hash,
            "sourceManifestHash": source_manifest_hash,
            "qualificationEligible": False,
            "issues": list(issue_values),
        }
        return cls(
            status,
            bool(passed),
            candidate_policy,
            source_binding,
            positions,
            policy_accuracy,
            expected_regret,
            illegal_action_rate,
            latency_p95_ms,
            baseline_policy,
            baseline_accuracy,
            baseline_expected_regret,
            evidence_hash,
            source_manifest_hash,
            False,
            issue_values,
            _canonical_hash(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "status": self.status,
            "passed": self.passed,
            "candidatePolicy": self.candidate_policy,
            "sourceBinding": self.source_binding,
            "positions": self.positions,
            "policyAccuracy": self.policy_accuracy,
            "expectedRegret": self.expected_regret,
            "illegalActionRate": self.illegal_action_rate,
            "latencyP95Ms": self.latency_p95_ms,
            "baselinePolicy": self.baseline_policy,
            "baselineAccuracy": self.baseline_accuracy,
            "baselineExpectedRegret": self.baseline_expected_regret,
            "evidenceHash": self.evidence_hash,
            "sourceManifestHash": self.source_manifest_hash,
            "qualificationEligible": self.qualification_eligible,
            "issues": list(self.issues),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FoundryAgentReleaseV1:
    release_id: str
    profile_id: str
    profile_hash: str
    profile: Mapping[str, Any]
    lineup_hash: str
    model_release_hash: str | None
    benchmark_profile_hash: str
    permanent_holdout_hash: str | None
    paired_comparison_hash: str | None
    symmetry_evidence_hash: str | None
    professional_review_hash: str | None
    configured_equal: bool
    zero_mistake_search_ready: bool
    benchmark_evidence_complete: bool
    professional_review_complete: bool
    expert_generation_allowed: bool
    status: str
    issues: tuple[str, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        release_id: str,
        profile: AgentProfile,
        lineup_snapshot: Mapping[str, Any],
        benchmark_profile: Mapping[str, Any],
        model_release_hash: str | None = None,
        permanent_holdout: Mapping[str, Any] | None = None,
        paired_comparison: Mapping[str, Any] | None = None,
        symmetry_evidence: Mapping[str, Any] | None = None,
        professional_review: Mapping[str, Any] | None = None,
    ) -> "FoundryAgentReleaseV1":
        identity = str(release_id).strip()
        if not identity:
            raise ValueError("Foundry agent release identity cannot be empty")
        snapshot = dict(lineup_snapshot)
        AgentProfileRegistry.from_snapshot(snapshot)
        lineup_hash = _sha256(snapshot.get("lineupHash"), "lineupHash")
        seats = dict(snapshot.get("seats") or {})
        configured_equal = (
            set(seats) == {"north", "west", "south", "east"}
            and {str(item.get("profileId")) for item in seats.values()} == {profile.id}
            and len({
                _canonical_hash(dict(item.get("tunings") or {}))
                for item in seats.values()
            }) == 1
        )
        profile_document = profile.to_dict()
        profile_hash = _canonical_hash(profile_document)
        issues: list[str] = []
        if not configured_equal:
            issues.append("unequal_lineup")

        static_checks = {
            "mistake_probability": profile.mistake_probability == 0.0,
            "deception_probability": profile.deception_probability == 0.0,
            "search_enabled": profile.play_search_mode == "paired_information_set",
            "search_budget": profile.play_search_budget > 0 and profile.play_search_samples >= 2,
            "bidding_evidence": profile.bid_samples >= 2,
            "signal_discipline": profile.signal_discipline >= 0.999999,
        }
        for name, passed in static_checks.items():
            if not passed:
                issues.append(f"static:{name}")
        zero_mistake_search_ready = configured_equal and all(static_checks.values())

        model_hash: str | None = None
        if profile.use_production_model:
            if model_release_hash is None:
                issues.append("model_release_missing")
            else:
                model_hash = _sha256(model_release_hash, "modelReleaseHash")
        elif model_release_hash is not None:
            model_hash = _sha256(model_release_hash, "modelReleaseHash")
        snapshot_model = dict(snapshot.get("modelSnapshot") or {})
        snapshot_model_hash = snapshot_model.get("modelReleaseHash") or snapshot_model.get("sha256")
        if model_hash is not None and snapshot_model_hash != model_hash:
            issues.append("model_snapshot_mismatch")

        benchmark_hash = _canonical_hash(dict(benchmark_profile))
        thresholds = dict(benchmark_profile.get("promotion_thresholds") or {})
        holdout_policy = dict(benchmark_profile.get("permanent_holdout") or {})
        paired_policy = dict(benchmark_profile.get("paired_comparison") or {})

        holdout_hash: str | None = None
        if permanent_holdout is None:
            issues.append("permanent_holdout_missing")
        else:
            holdout = dict(permanent_holdout)
            holdout_hash = _result_hash(holdout, "permanentHoldout.resultHash")
            archetypes = dict(holdout.get("archetypes") or {})
            holdout_checks = {
                "minimum_examples": int(holdout.get("examples", 0))
                >= max(
                    int(thresholds.get("minimum_examples", 1)),
                    int(holdout_policy.get("minimum_positions", 1)),
                ),
                "policy_accuracy": float(holdout.get("policyAccuracy", 0.0))
                >= float(thresholds.get("minimum_policy_accuracy", 1.0)),
                "expected_regret": float(holdout.get("expectedRegret", float("inf")))
                <= float(thresholds.get("maximum_expected_regret", 0.0)),
                "legal_actions": float(holdout.get("illegalActionRate", 1.0)) == 0.0,
                "archetype_coverage": all(
                    int(dict(archetypes.get(stratum) or {}).get("examples", 0))
                    >= int(holdout_policy.get("minimum_per_archetype", 1))
                    for stratum in benchmark_profile.get("strata") or ()
                ),
                "complete_labels": sum(
                    int(value) for value in dict(holdout.get("labelProvenance") or {}).values()
                ) == int(holdout.get("examples", -1)),
                "no_failed_search": not {
                    "failed", "cancelled"
                }.intersection({
                    key for key, count in dict(holdout.get("searchCompletion") or {}).items()
                    if int(count) > 0
                }),
            }
            for name, passed in holdout_checks.items():
                if not passed:
                    issues.append(f"holdout:{name}")

        pair_hash: str | None = None
        if paired_comparison is None:
            issues.append("paired_comparison_missing")
        else:
            paired = dict(paired_comparison)
            pair_hash = _result_hash(paired, "pairedComparison.resultHash")
            expected_candidate = str(paired_policy.get("candidate_profile_id") or "")
            expected_baseline = str(paired_policy.get("baseline_profile_id") or "")
            interval = tuple(float(item) for item in dict(paired.get("confidence95") or {}).get("decisiveWinRate", (0.0, 0.0)))
            pair_checks = {
                "candidate_identity": str(paired.get("candidateId") or "") == expected_candidate == profile.id,
                "baseline_identity": str(paired.get("baselineId") or "") == expected_baseline,
                "minimum_pairs": int(paired.get("pairs", 0)) >= int(paired_policy.get("minimum_pairs", 1)),
                "win_rate": float(paired.get("decisiveWinRate", 0.0)) >= float(paired_policy.get("minimum_win_rate", 1.0)),
                "win_rate_lower_bound": len(interval) == 2 and interval[0] > float(paired_policy.get("minimum_win_rate_lower_bound", 1.0)),
                "regret_improvement": float(paired.get("regretImprovement", 0.0)) >= float(paired_policy.get("minimum_regret_improvement", 1.0)),
                "promotion_target": paired.get("promotionTargetMet") is True,
            }
            for name, passed in pair_checks.items():
                if not passed:
                    issues.append(f"paired:{name}")

        symmetry_hash: str | None = None
        if symmetry_evidence is None:
            issues.append("symmetry_evidence_missing")
        else:
            symmetry = dict(symmetry_evidence)
            symmetry_hash = _result_hash(symmetry, "symmetryEvidence.resultHash")
            symmetry_policy = dict(benchmark_profile.get("symmetry") or {})
            required_transforms = set(symmetry_policy.get("required_transforms") or ())
            symmetry_valid = (
                symmetry.get("passed") is True
                and set(symmetry.get("transforms") or ()) == required_transforms
                and int(symmetry.get("comparedDecisions", 0)) > 0
                and float(symmetry.get("maximumDelta", float("inf")))
                <= float(symmetry_policy.get("tolerance", -1.0))
                and str(symmetry.get("profileId") or "") == profile.id
                and str(symmetry.get("profileHash") or "") == profile_hash
            )
            if not symmetry_valid:
                issues.append("symmetry_failed")

        review_hash: str | None = None
        if professional_review is None:
            issues.append("professional_review_missing")
        else:
            review = dict(professional_review)
            review_hash = _sha256(review.get("attestationHash"), "professionalReview.attestationHash")
            attestation_payload = {
                key: item for key, item in review.items() if key != "attestationHash"
            }
            expected_reviewed_hashes = {
                value for value in (holdout_hash, pair_hash, symmetry_hash) if value is not None
            }
            if (
                review_hash != _canonical_hash(attestation_payload)
                or review.get("approved") is not True
                or not str(review.get("reviewerId") or "").strip()
                or set(review.get("reviewedEvidenceHashes") or ())
                != expected_reviewed_hashes
            ):
                issues.append("professional_review_incomplete")
            if "professional_baloot" not in set(review.get("roles") or ()):
                issues.append("professional_review_role")

        benchmark_issues = [
            item for item in issues
            if item.startswith("holdout:")
            or item.startswith("paired:")
            or item in {
                "permanent_holdout_missing",
                "paired_comparison_missing",
                "symmetry_evidence_missing",
                "symmetry_failed",
                "model_release_missing",
                "model_snapshot_mismatch",
            }
        ]
        review_issues = [item for item in issues if item.startswith("professional_review")]
        benchmark_complete = zero_mistake_search_ready and not benchmark_issues
        professional_complete = not review_issues
        expert_allowed = benchmark_complete and professional_complete
        status = (
            "expert_qualified"
            if expert_allowed
            else "benchmark_qualified_pending_professional_review"
            if benchmark_complete
            else "configured_candidate"
            if zero_mistake_search_ready
            else "rejected_for_expert_generation"
        )
        core = {
            "schemaVersion": 1,
            "releaseId": identity,
            "profileId": profile.id,
            "profileHash": profile_hash,
            "profile": profile_document,
            "lineupHash": lineup_hash,
            "modelReleaseHash": model_hash,
            "benchmarkProfileHash": benchmark_hash,
            "permanentHoldoutHash": holdout_hash,
            "pairedComparisonHash": pair_hash,
            "symmetryEvidenceHash": symmetry_hash,
            "professionalReviewHash": review_hash,
            "configuredEqual": configured_equal,
            "zeroMistakeSearchReady": zero_mistake_search_ready,
            "benchmarkEvidenceComplete": benchmark_complete,
            "professionalReviewComplete": professional_complete,
            "expertGenerationAllowed": expert_allowed,
            "status": status,
            "issues": sorted(set(issues)),
        }
        return cls(
            identity,
            profile.id,
            profile_hash,
            profile_document,
            lineup_hash,
            model_hash,
            benchmark_hash,
            holdout_hash,
            pair_hash,
            symmetry_hash,
            review_hash,
            configured_equal,
            zero_mistake_search_ready,
            benchmark_complete,
            professional_complete,
            expert_allowed,
            status,
            tuple(core["issues"]),
            _canonical_hash(core),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "releaseId": self.release_id,
            "profileId": self.profile_id,
            "profileHash": self.profile_hash,
            "profile": dict(self.profile),
            "lineupHash": self.lineup_hash,
            "modelReleaseHash": self.model_release_hash,
            "benchmarkProfileHash": self.benchmark_profile_hash,
            "permanentHoldoutHash": self.permanent_holdout_hash,
            "pairedComparisonHash": self.paired_comparison_hash,
            "symmetryEvidenceHash": self.symmetry_evidence_hash,
            "professionalReviewHash": self.professional_review_hash,
            "configuredEqual": self.configured_equal,
            "zeroMistakeSearchReady": self.zero_mistake_search_ready,
            "benchmarkEvidenceComplete": self.benchmark_evidence_complete,
            "professionalReviewComplete": self.professional_review_complete,
            "expertGenerationAllowed": self.expert_generation_allowed,
            "status": self.status,
            "issues": list(self.issues),
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class FoundryAgentQualificationBundleV1:
    """Immutable, content-bound inputs and rebuilt agent-release decision."""

    release: FoundryAgentReleaseV1
    permanent_holdout: Mapping[str, Any]
    paired_comparison: Mapping[str, Any]
    symmetry_evidence: Mapping[str, Any]
    professional_review: Mapping[str, Any] | None
    sha256: str

    @classmethod
    def build(
        cls,
        *,
        release_id: str,
        profile: AgentProfile,
        lineup_snapshot: Mapping[str, Any],
        benchmark_profile: Mapping[str, Any],
        permanent_holdout: Mapping[str, Any],
        paired_comparison: Mapping[str, Any],
        symmetry_evidence: Mapping[str, Any],
        professional_review: Mapping[str, Any] | None = None,
    ) -> "FoundryAgentQualificationBundleV1":
        holdout = dict(permanent_holdout)
        paired = dict(paired_comparison)
        symmetry = dict(symmetry_evidence)
        review = dict(professional_review) if professional_review is not None else None
        release = FoundryAgentReleaseV1.build(
            release_id=release_id,
            profile=profile,
            lineup_snapshot=lineup_snapshot,
            benchmark_profile=benchmark_profile,
            permanent_holdout=holdout,
            paired_comparison=paired,
            symmetry_evidence=symmetry,
            professional_review=review,
        )
        core = {
            "schemaVersion": 1,
            "kind": "foundry_agent_qualification_bundle",
            "release": release.to_dict(),
            "evidence": {
                "permanentHoldout": holdout,
                "pairedComparison": paired,
                "symmetry": symmetry,
                "professionalReview": review,
            },
        }
        return cls(release, holdout, paired, symmetry, review, _canonical_hash(core))

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        profile: AgentProfile,
        lineup_snapshot: Mapping[str, Any],
        benchmark_profile: Mapping[str, Any],
    ) -> "FoundryAgentQualificationBundleV1":
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = _sha256(value.get("sha256"), "agentQualificationBundle.sha256")
        payload = {key: item for key, item in value.items() if key != "sha256"}
        if _canonical_hash(payload) != expected:
            raise ValueError("agent qualification bundle checksum mismatch")
        if value.get("kind") != "foundry_agent_qualification_bundle":
            raise ValueError("agent qualification bundle kind is invalid")
        release_value = dict(value.get("release") or {})
        evidence = dict(value.get("evidence") or {})
        rebuilt = cls.build(
            release_id=str(release_value.get("releaseId") or ""),
            profile=profile,
            lineup_snapshot=lineup_snapshot,
            benchmark_profile=benchmark_profile,
            permanent_holdout=dict(evidence.get("permanentHoldout") or {}),
            paired_comparison=dict(evidence.get("pairedComparison") or {}),
            symmetry_evidence=dict(evidence.get("symmetry") or {}),
            professional_review=(
                dict(evidence["professionalReview"])
                if evidence.get("professionalReview") is not None
                else None
            ),
        )
        if rebuilt.release.to_dict() != release_value or rebuilt.sha256 != expected:
            raise ValueError("agent qualification bundle does not rebuild exactly")
        return rebuilt

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "kind": "foundry_agent_qualification_bundle",
            "release": self.release.to_dict(),
            "evidence": {
                "permanentHoldout": dict(self.permanent_holdout),
                "pairedComparison": dict(self.paired_comparison),
                "symmetry": dict(self.symmetry_evidence),
                "professionalReview": (
                    dict(self.professional_review)
                    if self.professional_review is not None
                    else None
                ),
            },
            "sha256": self.sha256,
        }

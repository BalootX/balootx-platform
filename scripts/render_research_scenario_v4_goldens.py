# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

if TYPE_CHECKING:
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


GOLDENS = (
    (
        "sun-hundred-no-salvage",
        ROOT / "examples/research-scenario-v4-golden.json",
    ),
    (
        "sun-hundred-unique-saving-move",
        ROOT / "examples/research-scenario-v4-unique-saving.json",
    ),
    (
        "sun-hundred-ace-sacrifice-two-aces",
        ROOT / "examples/research-scenario-v4-ace-sacrifice-two-aces.json",
    ),
)
AUDIT_GOLDEN = (
    "sun-hundred-ace-sacrifice-rejected-audit",
    ROOT / "examples/research-scenario-v4-rejected-audit-media.json",
)


def _bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_result_code(scenario_template: str, exact_state_id: str) -> str:
    from baloot_platform.research.scenario_contracts import canonical_sha256

    digest = canonical_sha256({
        "seedDerivationVersion": 2,
        "scenarioTemplate": scenario_template,
        "exactStateId": exact_state_id,
    })
    return f"LAF-{digest[:10].upper()}"


def build_golden_bundle(
    *,
    name: str,
    fixture_path: Path,
    rules: RuleBook,
    renderer: RemotionProfessionalRenderer,
    video_profile_hash: str,
    export_preset: str = "hooked_flagship",
    trusted_complete_trace: dict | None = None,
):
    from baloot_platform.core.exact_state_view import (
        build_observer_safe_exact_state_v1,
    )
    from baloot_platform.research.exact_state import ExactStateNormalizerV1
    from baloot_platform.research.proof_service import UnifiedScenarioProofServiceV1
    from baloot_platform.research.render_service import (
        ResearchRenderPreparationServiceV1,
    )
    from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    normalized = ExactStateNormalizerV1(rules).normalize(fixture["exactTracePrefix"])
    if normalized.next_actor is None:
        raise ValueError(f"golden scenario {name} has no acting observer")
    proof = (
        UnifiedScenarioProofServiceV1.from_repository(ROOT)
        .prove(
            normalized,
            ScenarioProofRequestV1.build(fixture["proof"]),
            bundle_hash=video_profile_hash,
        )
        .to_dict()
    )
    expected = dict(fixture["expected"])
    if (
        proof["completionStatus"] != "exact_complete"
        or proof["winningRootActionCount"] != expected["winningRootActionCount"]
    ):
        raise ValueError(
            f"golden scenario {name} proof no longer matches its authority"
        )
    projection = {
        "schemaVersion": 1,
        "scenarioTemplate": name,
        "origin": "exact_trace_prefix",
        "status": "satisfied",
        "decisionKind": normalized.decision_kind,
        "requestedProofMode": proof["requestedMode"],
        "achievedProofMode": proof["achievedMode"],
        "proofCompletion": proof["completionStatus"],
        "winningRootActionCount": proof["winningRootActionCount"],
        "exactStateAvailable": True,
        "exactStateId": normalized.exact_state_id,
    }
    exact_state = normalized.to_dict()
    publication_eligible, disposition, failures, blockers, policy = (
        ResearchRenderPreparationServiceV1._scenario_publication_governance(
            {
                "disposition": "qualified",
                "qualification": {"qualified": True, "failures": []},
                "scenarioProjection": projection,
                "truthLabel": "exact_optimal",
                "score": 100.0,
                "confidence": 1.0,
            },
            proof=proof,
            exact_state_id=normalized.exact_state_id,
            video_profile_hash=video_profile_hash,
        )
    )
    if not publication_eligible:
        raise ValueError(f"golden scenario {name} is not publication eligible")
    result_code = _fixture_result_code(name, normalized.exact_state_id)
    decision_reference = {"resultCode": result_code, "decisionOrdinal": 1}
    public_exact = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=projection,
        observer_seat=normalized.next_actor,
        decision_reference=decision_reference,
    )
    projection_for_render = {
        **projection,
        "decisionReference": decision_reference,
        "publicProjectionHash": public_exact["publicProjectionHash"],
    }
    public_policy = (
        ResearchRenderPreparationServiceV1._public_scenario_publication_policy(
            policy, result_code=result_code
        )
    )
    source = {
        "sourceSchemaVersion": 3,
        "kind": "research_scenario_result",
        "code": result_code,
        "decisionOrdinal": 1,
        "decisionReference": decision_reference,
        "publicProjectionHash": public_exact["publicProjectionHash"],
        "qualificationStatus": disposition,
        "renderPurpose": "content_candidate",
        "qualificationFailures": failures,
        "publicationBlockers": blockers,
        "observerSeat": normalized.next_actor,
        "auditOnly": False,
        "publicationEligible": publication_eligible,
        "qrActionable": False,
        "publicationPolicy": public_policy,
        "rulesHash": rules.sha256,
        "proofReference": result_code,
        "proofCompletionStatus": proof["completionStatus"],
        "proofClaimSatisfied": proof["claimSatisfied"],
        "proofAchievedMode": proof["achievedMode"],
    }
    return renderer.build_scenario_render_request(
        rules=rules,
        exact_state=exact_state,
        proof_outcome=proof,
        scenario_projection=projection_for_render,
        effective_config_sha256=video_profile_hash,
        game_serial=result_code,
        observer_seat=normalized.next_actor,
        enrollment_uri=f"balootx://golden/results/{result_code}/decisions/1",
        export_preset=export_preset,
        source_metadata=source,
        trusted_complete_trace=trusted_complete_trace,
    )


def build_audit_bundle(
    *,
    name: str,
    fixture_path: Path,
    rules: RuleBook,
    renderer: RemotionProfessionalRenderer,
    video_profile_hash: str,
    export_preset: str = "hooked_flagship",
):
    from baloot_platform.core.exact_state_view import (
        build_observer_safe_exact_state_v1,
    )
    from baloot_platform.research.exact_state import ExactStateNormalizerV1
    from baloot_platform.research.proof_service import UnifiedScenarioProofServiceV1
    from baloot_platform.research.render_service import (
        ResearchRenderPreparationServiceV1,
    )
    from baloot_platform.research.scenario_contracts import (
        ScenarioProofRequestV1,
        canonical_sha256,
    )

    audit_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schemaVersion",
        "kind",
        "fixtureId",
        "baseFixture",
        "proofOverrides",
        "resultDisposition",
        "expected",
    }
    if (
        set(audit_fixture) != expected_keys
        or audit_fixture["schemaVersion"] != 1
        or audit_fixture["kind"]
        != "research_scenario_v4_rejected_audit_media"
    ):
        raise ValueError("Research V4 audit-media fixture is invalid")
    base_relative = PurePosixPath(str(audit_fixture["baseFixture"]))
    base_path = ROOT.joinpath(*base_relative.parts).resolve()
    allowed_bases = {path.resolve() for _, path in GOLDENS}
    if base_path not in allowed_bases:
        raise ValueError("Research V4 audit-media base fixture is not governed")
    base_fixture = json.loads(base_path.read_text(encoding="utf-8"))
    proof_overrides = dict(audit_fixture["proofOverrides"])
    if set(proof_overrides) != {"selectedAction"}:
        raise ValueError("Research V4 audit-media proof override is invalid")

    normalized = ExactStateNormalizerV1(rules).normalize(
        base_fixture["exactTracePrefix"]
    )
    if normalized.next_actor is None:
        raise ValueError(f"audit scenario {name} has no acting observer")
    proof_request = {**dict(base_fixture["proof"]), **proof_overrides}
    proof = (
        UnifiedScenarioProofServiceV1.from_repository(ROOT)
        .prove(
            normalized,
            ScenarioProofRequestV1.build(proof_request),
            bundle_hash=video_profile_hash,
        )
        .to_dict()
    )
    expected = dict(audit_fixture["expected"])
    for key in (
        "completionStatus",
        "claimSatisfied",
        "selectedAction",
        "winningRootActionCount",
    ):
        if proof.get(key) != expected[key]:
            raise ValueError(
                f"audit scenario {name} proof no longer matches {key}"
            )
    projection = {
        "schemaVersion": 1,
        "scenarioTemplate": str(audit_fixture["fixtureId"]),
        "origin": "exact_trace_prefix",
        "status": proof["completionStatus"],
        "decisionKind": normalized.decision_kind,
        "requestedProofMode": proof["requestedMode"],
        "achievedProofMode": proof["achievedMode"],
        "proofCompletion": proof["completionStatus"],
        "winningRootActionCount": proof["winningRootActionCount"],
        "exactStateAvailable": True,
        "exactStateId": normalized.exact_state_id,
    }
    exact_state = normalized.to_dict()
    result = dict(audit_fixture["resultDisposition"])
    result["scenarioProjection"] = projection
    result_hash = canonical_sha256(
        {
            "auditFixture": audit_fixture,
            "baseFixture": base_fixture,
            "exactStateHash": exact_state["sha256"],
            "proofHash": proof["semanticSha256"],
            "scenarioProjection": projection,
        }
    )
    publication_eligible, disposition, failures, blockers, policy = (
        ResearchRenderPreparationServiceV1._scenario_publication_governance(
            result,
            proof=proof,
            exact_state_id=normalized.exact_state_id,
            video_profile_hash=video_profile_hash,
        )
    )
    if publication_eligible:
        raise ValueError(f"audit scenario {name} became publication eligible")
    if (
        disposition != expected["qualificationStatus"]
        or expected["publicationEligible"] is not False
        or expected["renderPurpose"] != "research_audit"
    ):
        raise ValueError(f"audit scenario {name} disposition no longer matches")
    result_code = _fixture_result_code(projection["scenarioTemplate"], normalized.exact_state_id)
    decision_reference = {"resultCode": result_code, "decisionOrdinal": 1}
    public_exact = build_observer_safe_exact_state_v1(
        exact_state=exact_state,
        scenario_projection=projection,
        observer_seat=normalized.next_actor,
        decision_reference=decision_reference,
    )
    projection_for_render = {
        **projection,
        "decisionReference": decision_reference,
        "publicProjectionHash": public_exact["publicProjectionHash"],
    }
    public_policy = (
        ResearchRenderPreparationServiceV1._public_scenario_publication_policy(
            policy, result_code=result_code
        )
    )
    source = {
        "sourceSchemaVersion": 3,
        "kind": "research_scenario_result",
        "code": result_code,
        "decisionOrdinal": 1,
        "decisionReference": decision_reference,
        "publicProjectionHash": public_exact["publicProjectionHash"],
        "qualificationStatus": disposition,
        "renderPurpose": "research_audit",
        "qualificationFailures": failures,
        "publicationBlockers": blockers,
        "observerSeat": normalized.next_actor,
        "auditOnly": True,
        "publicationEligible": False,
        "qrActionable": False,
        "publicationPolicy": public_policy,
        "rulesHash": rules.sha256,
        "proofReference": result_code,
        "proofCompletionStatus": proof["completionStatus"],
        "proofClaimSatisfied": proof["claimSatisfied"],
        "proofAchievedMode": proof["achievedMode"],
    }
    bundle = renderer.build_scenario_render_request(
        rules=rules,
        exact_state=exact_state,
        proof_outcome=proof,
        scenario_projection=projection_for_render,
        effective_config_sha256=video_profile_hash,
        game_serial=result_code,
        observer_seat=normalized.next_actor,
        enrollment_uri=f"balootx://audit/results/{result_code}/decisions/1",
        export_preset=export_preset,
        source_metadata=source,
    )
    receipt_core = {
        "schemaVersion": 1,
        "kind": "research_scenario_v4_audit_media_preparation",
        "fixture": fixture_path.relative_to(ROOT).as_posix(),
        "baseFixture": base_path.relative_to(ROOT).as_posix(),
        "resultHash": result_hash,
        "exactStateId": normalized.exact_state_id,
        "proofOutcome": proof,
        "scenarioProjection": projection,
        "source": source,
        "bundleSha256": bundle.bundle_sha256,
    }
    receipt = {**receipt_core, "sha256": canonical_sha256(receipt_core)}
    return bundle, receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and render the governed Research V4 golden scenarios."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/research-scenario-v4-goldens",
    )
    parser.add_argument("--bundle-only", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Prepare only the governed rejected V4 audit-media fixture.",
    )
    parser.add_argument(
        "--export-preset",
        choices=("hooked_flagship", "decision_clip"),
        default="hooked_flagship",
    )
    args = parser.parse_args()
    from baloot_platform.core.rules import RuleBook
    from baloot_platform.research.scenario_contracts import canonical_sha256
    from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rules = RuleBook.load(ROOT / "configs/rules/saudi_common_v1.yaml")
    profile = yaml.safe_load(
        (ROOT / "configs/video/premium.yaml").read_text(encoding="utf-8")
    )
    profile_hash = canonical_sha256(profile)
    renderer = RemotionProfessionalRenderer(profile, repository_root=ROOT)
    entries: list[dict[str, Any]] = []
    jobs = (AUDIT_GOLDEN,) if args.audit_only else GOLDENS
    for name, fixture_path in jobs:
        receipt = None
        if args.audit_only:
            bundle, receipt = build_audit_bundle(
                name=name,
                fixture_path=fixture_path,
                rules=rules,
                renderer=renderer,
                video_profile_hash=profile_hash,
                export_preset=args.export_preset,
            )
        else:
            bundle = build_golden_bundle(
                name=name,
                fixture_path=fixture_path,
                rules=rules,
                renderer=renderer,
                video_profile_hash=profile_hash,
                export_preset=args.export_preset,
            )
        bundle_path = output_dir / f"{name}-render-bundle.json"
        _write_json(bundle_path, bundle.to_dict())
        entry: dict[str, Any] = {
            "name": name,
            "fixture": fixture_path.relative_to(ROOT).as_posix(),
            "fixtureSha256": _sha256(fixture_path),
            "bundle": bundle_path.name,
            "bundleSha256": bundle.bundle_sha256,
            "bundleFileSha256": _sha256(bundle_path),
            "decisionReference": bundle.value["source"]["decisionReference"],
            "publicProjectionHash": bundle.value["source"][
                "publicProjectionHash"
            ],
            "exportPreset": args.export_preset,
        }
        if receipt is not None:
            receipt_path = output_dir / f"{name}-preparation.json"
            _write_json(receipt_path, receipt)
            entry.update(
                {
                    "preparation": receipt_path.name,
                    "preparationSha256": _sha256(receipt_path),
                    "qualificationStatus": bundle.value["source"][
                        "qualificationStatus"
                    ],
                    "renderPurpose": bundle.value["source"]["renderPurpose"],
                    "auditOnly": bundle.value["source"]["auditOnly"],
                    "publicationEligible": bundle.value["source"][
                        "publicationEligible"
                    ],
                }
            )
        if not args.bundle_only:
            artifacts = renderer.execute(
                bundle,
                export_preset=args.export_preset,
                output_dir=output_dir,
                basename=name,
            )
            entry["renderManifest"] = artifacts.manifest.name
            entry["renderManifestSha256"] = _sha256(artifacts.manifest)
            entry["video"] = artifacts.video.name
            entry["videoSha256"] = _sha256(artifacts.video)
        entries.append(entry)
    manifest = {
        "schemaVersion": 1,
        "kind": (
            "research_scenario_v4_rejected_audit_media"
            if args.audit_only
            else "research_scenario_v4_golden_media"
        ),
        "rulesHash": rules.sha256,
        "videoProfileHash": profile_hash,
        "bundleOnly": bool(args.bundle_only),
        "exportPreset": args.export_preset,
        "entries": entries,
    }
    manifest["manifestSha256"] = canonical_sha256(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

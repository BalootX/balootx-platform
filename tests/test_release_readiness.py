from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from baloot_platform.benchmark import canonical_sha256
from scripts import generate_security_report as security_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check_release_readiness.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_readiness", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def test_exact_evidence_is_integrity_and_state_bound(tmp_path: Path) -> None:
    module = _load()
    candidates = module.load_candidate_jsonl(module.DEFAULT_CANDIDATES)
    candidate = candidates["positions"][0]
    value = {
        "schemaVersion": 1,
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "oracleId": "test-exact-oracle",
        "oracleAuthoritySha256": "a" * 64,
        "exactLabelCount": 1,
        "positions": [
            {
                "candidateId": candidate["candidateId"],
                "stateSha256": candidate["stateSha256"],
                "observer": candidate["observer"],
                "legalActions": candidate["legalActions"],
                "searchCompletion": "exact",
            }
        ],
    }
    value["evidenceSha256"] = canonical_sha256(value)
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = module._exact_gate(path, candidates)

    assert result["status"] == "passed"
    assert result["exactLabelCount"] == 1


def test_missing_external_adjudication_is_pending_not_approved(tmp_path: Path) -> None:
    module = _load()
    candidates = module.load_candidate_jsonl(module.DEFAULT_CANDIDATES)
    result = module._holdout_gate(
        tmp_path / "missing.jsonl",
        candidates,
        minimum_positions=500,
        minimum_authorities=2,
        minimum_agreement=0.8,
    )

    assert result["status"] == "pending"
    assert "not been finalized" in result["reason"]


def test_source_gate_ignores_evidence_but_rejects_product_changes(
    tmp_path: Path,
) -> None:
    module = _load()
    tracked = (
        "release/evidence.json",
        "src/product.py",
        "configs/rules.yaml",
        "schemas/rules.json",
        "generated/openapi.json",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    for relative in tracked:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n", encoding="utf-8", newline="\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=BalootX Test",
            "-c",
            "user.email=balootx-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    (tmp_path / "release/evidence.json").write_text(
        "generated\n", encoding="utf-8", newline="\n"
    )
    evidence_only = module._source_gate(tmp_path)
    assert evidence_only["status"] == "passed"
    assert evidence_only["clean"] is True
    assert evidence_only["scope"] == "git_visible_product_inputs_v1"
    assert evidence_only["excludedPaths"] == [
        "build/",
        "dist/",
        "output/",
        "outputs/",
        "release/",
    ]

    for relative in tracked[1:]:
        path = tmp_path / relative
        path.write_text("changed\n", encoding="utf-8", newline="\n")
        product_change = module._source_gate(tmp_path)
        assert product_change["status"] == "pending", relative
        assert product_change["clean"] is False
        subprocess.run(["git", "restore", relative], cwd=tmp_path, check=True)


def test_holdout_gate_rejects_forged_manifest(tmp_path: Path) -> None:
    module = _load()
    candidates = module.load_candidate_jsonl(module.DEFAULT_CANDIDATES)
    value = {
        "schemaVersion": 1,
        "datasetId": "forged",
        "datasetSha256": canonical_sha256([]),
        "candidateCount": 0,
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "positionCount": 0,
        "agreementRate": 1.0,
        "reviewerIds": ["a", "b"],
        "authorityIds": ["x", "y"],
        "labelProvenance": {"exact": 0, "independentlyAdjudicated": 0},
        "positions": [],
        "automaticPromotion": False,
        "manifestSha256": "0" * 64,
    }
    path = tmp_path / "forged.jsonl"
    path.write_text(
        json.dumps({key: item for key, item in value.items() if key != "positions"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = module._holdout_gate(
        path,
        candidates,
        minimum_positions=500,
        minimum_authorities=2,
        minimum_agreement=0.8,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "holdout manifest hash mismatch"


def test_holdout_gate_accepts_integrity_bound_adjudication(tmp_path: Path) -> None:
    module = _load()
    candidates = module.load_candidate_jsonl(module.DEFAULT_CANDIDATES)
    positions = []
    for index, candidate in enumerate(candidates["positions"][:500]):
        action = candidate["legalActions"][0]
        source = "exact" if index < 120 else "independent_adjudication"
        positions.append(
            {
                "positionId": candidate["candidateId"],
                "stateSha256": candidate["stateSha256"],
                "observer": candidate["observer"],
                "legalActions": candidate["legalActions"],
                "referenceAction": action,
                "referenceActions": [action],
                "labelSource": source,
            }
        )
    value = {
        "schemaVersion": 1,
        "datasetId": "permanent-holdout-test",
        "candidateDatasetId": candidates["datasetId"],
        "candidateDatasetSha256": candidates["datasetSha256"],
        "positionCount": len(positions),
        "agreementRate": 1.0,
        "disagreements": 0,
        "reviewerIds": ["reviewer-a", "reviewer-b"],
        "authorityIds": ["authority-a", "authority-b"],
        "labelProvenance": {"exact": 120, "independentlyAdjudicated": 380},
        "automaticPromotion": False,
        "positions": positions,
    }
    value["datasetSha256"] = canonical_sha256(positions)
    value["manifestSha256"] = canonical_sha256(value)
    path = tmp_path / "holdout.jsonl"
    header = {key: item for key, item in value.items() if key != "positions"}
    path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in [header, *positions]),
        encoding="utf-8",
    )

    result = module._holdout_gate(
        path,
        candidates,
        minimum_positions=500,
        minimum_authorities=2,
        minimum_agreement=0.8,
    )

    assert result["status"] == "passed"
    assert result["positionCount"] == 500
    assert result["labelProvenance"] == {
        "exact": 120,
        "independentlyAdjudicated": 380,
    }


def test_media_gate_requires_only_vertical_release_artifacts(tmp_path: Path) -> None:
    module = _load()
    result = module._media_gate(tmp_path, 1080, 1920)

    assert result["status"] == "pending"
    assert result["requiredAspectRatio"] == "9:16"
    assert result["forbiddenReleaseAspectRatios"] == ["16:9", "1:1"]
    assert all("v21" not in path.lower() for path in result["missing"])


def test_media_gate_rejects_non_vertical_probe(tmp_path: Path) -> None:
    module = _load()
    for relative in module.REQUIRED_RELEASE_MEDIA:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    probe = tmp_path / "release/media-probe.json"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "status": "passed",
                "media": [
                    {
                        "path": "bad.mp4",
                        "mediaProbe": {"video": {"width": 1920, "height": 1080}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = module._media_gate(tmp_path, 1080, 1920)

    assert result["status"] == "failed"
    assert result["invalidAspectRatio"] == ["bad.mp4"]


def test_media_gate_accepts_canonical_vertical_probe(tmp_path: Path) -> None:
    module = _load()
    for relative in module.REQUIRED_RELEASE_MEDIA:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    probe = tmp_path / "release/media-probe.json"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "status": "passed",
                "media": [
                    {
                        "path": "flagship.mp4",
                        "mediaProbe": {"video": {"width": 1080, "height": 1920}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = module._media_gate(tmp_path, 1080, 1920)

    assert result["status"] == "passed"
    assert result["invalidAspectRatio"] == []


def test_release_evidence_rejects_present_but_pending_artifact(
    tmp_path: Path,
) -> None:
    module = _load()
    path = tmp_path / "release/visual-qa.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "pending"}), encoding="utf-8")

    result = module._evidence_artifact(
        tmp_path,
        "release/visual-qa.json",
    )

    assert result["status"] == "failed"
    assert "accepted evidence status" in result["reason"]


def test_release_evidence_accepts_only_current_unsigned_rules_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    bundle = SimpleNamespace(
        sha256="a" * 64,
        rules=SimpleNamespace(sha256="b" * 64),
    )
    monkeypatch.setattr(module, "build_bundle", lambda _root: bundle)
    path = tmp_path / "release/rules-certification-candidate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "status": "passed",
        "official": False,
        "automaticRuleActivation": False,
        "certificationStatus": "candidate_external_review_required",
        "bundleSha256": bundle.sha256,
        "rulesSha256": bundle.rules.sha256,
        "unresolvedTraceabilityClauses": [],
    }
    value["candidateSha256"] = module.canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    result = module._evidence_artifact(
        tmp_path,
        "release/rules-certification-candidate.json",
    )

    assert result["status"] == "passed"

    stale = dict(value)
    stale["bundleSha256"] = "c" * 64
    stale["candidateSha256"] = module.canonical_sha256(
        {key: item for key, item in stale.items() if key != "candidateSha256"}
    )
    path.write_text(json.dumps(stale), encoding="utf-8")

    assert module._evidence_artifact(
        tmp_path,
        "release/rules-certification-candidate.json",
    )["status"] == "failed"


def test_release_evidence_binds_intelligence_benchmark_to_current_authorities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    bundle = SimpleNamespace(rules=SimpleNamespace(sha256="a" * 64))
    monkeypatch.setattr(module, "build_bundle", lambda _root: bundle)
    monkeypatch.setattr(
        module,
        "benchmark_configuration_sha256",
        lambda _bundle: "b" * 64,
    )
    monkeypatch.setattr(
        module,
        "load_candidate_jsonl",
        lambda _path: {"datasetId": "current", "datasetSha256": "c" * 64},
    )
    release = tmp_path / "release"
    release.mkdir(parents=True)
    (release / "exact-labels.json").write_text(
        json.dumps({"evidenceSha256": "d" * 64}),
        encoding="utf-8",
    )
    source_manifest = {
        "schemaVersion": 1,
        "source": {
            "revision": "e" * 40,
            "worktreeSnapshot": {"sha256": "f" * 64},
        },
    }
    source_path = release / "source-manifest.json"
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    value = {
        "status": "passed",
        "kind": "observer_safe_exact_subset_preanalysis",
        "candidateDatasetId": "current",
        "candidateDatasetSha256": "c" * 64,
        "exactEvidenceSha256": "d" * 64,
        "rulesSha256": "a" * 64,
        "configurationSha256": "b" * 64,
        "source": {
            "binding": "local_workspace_manifest",
            "revision": "e" * 40,
            "tree": None,
            "workspaceManifestSha256": "f" * 64,
        },
        "policies": {"strong": {"top1Accuracy": 1.0, "latencyMs": {}}},
        "automaticPromotion": False,
        "promotionDecision": {"decision": "not_eligible"},
    }
    value["semanticResultSha256"] = module.canonical_sha256(
        {
            "candidateDatasetSha256": value["candidateDatasetSha256"],
            "exactEvidenceSha256": value["exactEvidenceSha256"],
            "policies": {"strong": {"top1Accuracy": 1.0}},
        }
    )
    path = release / "intelligence-benchmark.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    assert module._evidence_artifact(
        tmp_path,
        "release/intelligence-benchmark.json",
    )["status"] == "passed"

    value["rulesSha256"] = "f" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    assert module._evidence_artifact(
        tmp_path,
        "release/intelligence-benchmark.json",
    )["status"] == "failed"


def test_release_evidence_rejects_resigned_stale_security_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load()
    source = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": "git_visible_product_inputs_v1",
        "excludedPaths": ["release/"],
        "binding": "git_head_plus_worktree_snapshot",
        "worktreeSnapshot": {
            "algorithm": "sha256",
            "scope": "git_visible_product_inputs_v1",
            "excludedPaths": ["release/"],
            "fileCount": 1,
            "statusSha256": "c" * 64,
            "sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(
        security_report,
        "git_identity",
        lambda root: source,
    )
    value = {
        "status": "passed",
        "source": source,
        "continuousIntegration": {
            "headSha": source["revision"],
            "conclusion": "success",
        },
    }
    value["evidenceSha256"] = canonical_sha256(value)
    monkeypatch.setattr(
        security_report,
        "build",
        lambda **_kwargs: value,
    )
    path = tmp_path / "release/security-report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")

    assert module._evidence_artifact(
        tmp_path,
        "release/security-report.json",
    )["status"] == "passed"

    stale = json.loads(path.read_text(encoding="utf-8"))
    stale["source"]["worktreeSnapshot"]["sha256"] = "e" * 64
    stale["evidenceSha256"] = canonical_sha256(
        {
            key: item
            for key, item in stale.items()
            if key != "evidenceSha256"
        }
    )
    path.write_text(json.dumps(stale), encoding="utf-8")
    rejected = module._evidence_artifact(
        tmp_path,
        "release/security-report.json",
    )
    assert rejected["status"] == "failed"
    assert "source identity is stale" in rejected["reason"]


def test_readiness_output_is_canonical_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    module = _load()
    monkeypatch.setattr(
        module,
        "_source_gate",
        lambda root: {
            "status": "passed",
            "gitCommit": "a" * 40,
            "gitTree": "b" * 40,
            "clean": True,
            "reason": None,
        },
    )
    monkeypatch.setattr(module, "REQUIRED_RELEASE_EVIDENCE", ())
    monkeypatch.setattr(
        module,
        "_media_gate",
        lambda root, width, height: {
            "status": "passed",
            "requiredAspectRatio": "9:16",
            "requiredResolution": {"width": width, "height": height},
            "artifacts": [],
        },
    )
    first = module.collect_readiness(
        ROOT,
        candidates_path=module.DEFAULT_CANDIDATES,
        exact_path=module.DEFAULT_EXACT,
        holdout_path=tmp_path / "missing.jsonl",
    )
    second = module.collect_readiness(
        ROOT,
        candidates_path=module.DEFAULT_CANDIDATES,
        exact_path=module.DEFAULT_EXACT,
        holdout_path=tmp_path / "missing.jsonl",
    )

    assert first == second
    assert first["status"] == "pending"
    assert first["manifestSha256"] == canonical_sha256(
        {key: item for key, item in first.items() if key != "manifestSha256"}
    )
    assert first["gates"]["independentAdjudication"]["status"] == "pending"
    assert first["gates"]["independentAdjudication"]["path"] == "missing.jsonl"
    assert first["gates"]["candidateCorpus"]["path"] == (
        "private-inputs/strength-candidates.jsonl"
    )
    assert first["gates"]["releaseMedia"]["requiredAspectRatio"] == "9:16"

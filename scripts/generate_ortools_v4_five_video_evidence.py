#!/usr/bin/env python3
"""Run one governed five-result V4 campaign and render its five selected videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OPERATOR_BUNDLE = "configs/bundle_intelligent_research.yaml"

from baloot_platform.research.contracts import canonical_sha256  # noqa: E402
from baloot_platform.config import build_bundle  # noqa: E402
from baloot_platform.core.models import Card  # noqa: E402
from baloot_platform.filesystem import atomic_write_text  # noqa: E402
from baloot_platform.research.recipe_v4 import ResearchRecipeV4  # noqa: E402
from baloot_platform.video.render_bundle import RenderBundleV1  # noqa: E402
from baloot_platform.video.render_plan import RenderBatchV1  # noqa: E402

try:  # noqa: E402
    from scripts.generate_performance_evidence import git_identity
except ModuleNotFoundError:  # direct script execution
    from generate_performance_evidence import git_identity  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_document_hash(value: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(value))


def _artifact_reference(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "release evidence must be stored inside the repository"
        ) from exc
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _persist_immutable_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    payload = (
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() != payload:
        raise ValueError(
            f"immutable Research artifact already exists with other bytes: {path}"
        )
    path.write_bytes(payload)
    return _artifact_reference(path, repository_root)


def persist_research_documents(
    output_root: Path,
    recipe: ResearchRecipeV4,
    results: list[Mapping[str, Any]],
    *,
    repository_root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    recipe_artifact = _persist_immutable_json(
        output_root / "recipes" / f"{recipe.code}.json",
        recipe.to_dict(),
        repository_root=repository_root,
    )
    result_artifacts: dict[str, dict[str, Any]] = {}
    for result in sorted(results, key=lambda item: str(item.get("resultCode") or "")):
        code = str(result.get("resultCode") or "")
        if not re.fullmatch(r"LAF-[A-Z0-9]+", code):
            raise ValueError(
                "Research result code is unsafe for immutable artifact storage"
            )
        result_artifacts[code] = _persist_immutable_json(
            output_root / "results" / f"{code}.json",
            result,
            repository_root=repository_root,
        )
    if len(result_artifacts) != 5:
        raise ValueError("five-video evidence requires five distinct result artifacts")
    return recipe_artifact, result_artifacts


def validate_recipe(value: Mapping[str, Any]) -> ResearchRecipeV4:
    recipe = ResearchRecipeV4.build(value)
    if recipe.scenario.origin.kind != "search":
        raise ValueError("five-video evidence requires a search-origin V4 recipe")
    if recipe.status != "published":
        raise ValueError("five-video evidence requires a published recipe")
    if recipe.output != {"kind": "laffa_collection", "resultCount": 5}:
        raise ValueError("five-video evidence requires exactly five requested results")
    if recipe.render_policy != {"mode": "on_demand", "videoCount": 5}:
        raise ValueError("five-video evidence requires exactly five on-demand videos")
    return recipe


def ensure_operator_bundle(container: Any, bundle_revision_id: str | None) -> str:
    service = container.configuration
    if bundle_revision_id:
        try:
            service.resolve(revision_id=bundle_revision_id)
            return bundle_revision_id
        except KeyError:
            pass
    service.bootstrap_packaged_bundle(actor_id="FIVE-VIDEO-EVIDENCE")
    active = service.resolve()
    active_profiles = dict(active["profiles"])
    active_revisions = dict(active["profileRevisions"])
    alternate = yaml.safe_load((ROOT / OPERATOR_BUNDLE).read_text(encoding="utf-8"))
    packaged = build_bundle(ROOT, bundle_path=OPERATOR_BUNDLE)
    components: dict[str, str] = {}
    for family in packaged.components:
        payload = (
            dict(packaged.rules.data)
            if family == "rules"
            else dict(packaged.components[family].data)
        )
        if active_profiles.get(family) == payload:
            components[family] = str(active_revisions[family]["revisionId"])
            continue
        profile = service.create_profile(
            family=family,
            payload=payload,
            created_by="FIVE-VIDEO-EVIDENCE",
            publish=True,
        )
        components[family] = str(profile["revisionId"])
    created = service.create_bundle(
        bundle_id=str(alternate["profile"]["id"]),
        version=str(alternate["profile"]["version"]),
        mode="full",
        component_revisions=components,
        created_by="FIVE-VIDEO-EVIDENCE",
        publish=True,
        allow_historical_components=True,
    )
    created_revision_id = str(created["revisionId"])
    if bundle_revision_id and created_revision_id != bundle_revision_id:
        raise ValueError(
            "standalone evidence bundle identity does not match the recipe: "
            f"expected {bundle_revision_id}, created {created['revisionId']}"
        )
    return created_revision_id


def _load_bundle(artifact_store: Any, reference: Mapping[str, Any]) -> RenderBundleV1:
    payload = artifact_store.get(str(reference["uri"]))
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("prepared render bundle artifact hash mismatch")
    bundle = RenderBundleV1(json.loads(raw))
    if not bundle.verify() or bundle.bundle_sha256 != reference.get("ownerHash"):
        raise ValueError("prepared render bundle identity mismatch")
    return bundle


def _validate_playback(manifest: Mapping[str, Any]) -> None:
    rows = manifest.get("playbackEvidence")
    files = manifest.get("files")
    if not isinstance(rows, list) or len(rows) not in (33, 34) or [
        row.get("completedPlayCount") for row in rows[:33]
    ] != list(range(33)):
        raise ValueError("video does not bind every play boundary")
    if not isinstance(files, Mapping):
        raise ValueError("video playback files are unavailable")
    if len(rows) == 34:
        outcome = rows[-1]
        if (outcome.get("checkpointPhase") != "outcome"
                or outcome.get("completedPlayCount") != 32
                or set(dict(outcome.get("handCounts") or {}).values()) != {0}
                or outcome.get("observerHandCardIds") != []
                or outcome.get("currentTrickCardIds") != []
                or outcome.get("logicalTableCardIds", outcome.get("tableCardIds")) != []
                or dict(outcome.get("encodedFrameComparison") or {}).get("frame") != outcome.get("frame")
                or dict(files.get("playback:outcome") or {}).get("sha256") != outcome.get("sourceSha256")):
            raise ValueError("video playback outcome checkpoint is invalid")
        rows = rows[:33]
    observer = str(dict(manifest.get("source") or {}).get("observerSeat") or "")
    if observer not in {"north", "west", "south", "east"}:
        raise ValueError("video playback observer seat is invalid")
    previous = {seat: 8 for seat in ("north", "west", "south", "east")}
    previous_observer_cards = list(rows[0].get("observerHandCardIds") or ())
    if (
        len(previous_observer_cards) != 8
        or len(set(previous_observer_cards)) != 8
        or any(type(card_id) is not int for card_id in previous_observer_cards)
    ):
        raise ValueError("video playback observer hand is invalid")
    played_card_ids: list[int] = []
    observer_played_ids: list[int] = []
    for count, row in enumerate(rows):
        counts = dict(row.get("handCounts") or {})
        if set(counts) != set(previous) or any(
            type(value) is not int for value in counts.values()
        ):
            raise ValueError("video playback hand-count evidence is invalid")
        if count:
            played = dict(row.get("playedCard") or {})
            actor = str(played.get("seat") or "")
            card_id = dict(played.get("card") or {}).get("id")
            if (
                actor not in previous
                or type(card_id) is not int
                or card_id in played_card_ids
            ):
                raise ValueError("video playback played-card evidence is invalid")
            played_card_ids.append(card_id)
            expected = {
                seat: value - (seat == actor) for seat, value in previous.items()
            }
            if counts != expected:
                raise ValueError(
                    "video playback does not remove exactly one played card"
                )
            expected_observer_cards = list(previous_observer_cards)
            if actor == observer:
                if card_id not in expected_observer_cards:
                    raise ValueError("video playback reveals an invalid observer card")
                expected_observer_cards.remove(card_id)
                observer_played_ids.append(card_id)
            if row.get("observerHandCardIds") != expected_observer_cards:
                raise ValueError(
                    "video playback observer hand does not follow the trace"
                )
            previous_observer_cards = expected_observer_cards
        position = count % 4
        expected_current = played_card_ids[-position:] if position else []
        expected_table = (
            played_card_ids[-4:] if count and not position else expected_current
        )
        if row.get("currentTrickCardIds") != expected_current:
            raise ValueError(
                "video playback current trick does not grow and clear correctly"
            )
        # Historical tableCardIds also described logical state, not animation visibility.
        table_cards = row.get("logicalTableCardIds", row.get("tableCardIds"))
        if "logicalTableCardIds" in row and "tableCardIds" in row:
            raise ValueError("video playback has competing table-state evidence")
        if table_cards != expected_table:
            raise ValueError("video playback table cards do not match the trace")
        expected_completed = count // 4 - 1 if count and not position else None
        if row.get("completedTrickIndex") != expected_completed:
            raise ValueError("video playback completed trick index is invalid")
        if len(row.get("observerHandCardIds") or ()) != counts[observer]:
            raise ValueError("video playback observer hand count is invalid")
        comparison = row.get("encodedFrameComparison")
        if not isinstance(comparison, Mapping) or int(
            comparison.get("frame", -1)
        ) != int(row.get("frame", -2)):
            raise ValueError(
                "video playback boundary lacks encoded-frame parity evidence"
            )
        source_file = files.get(f"playback:{count:02d}")
        if not isinstance(source_file, Mapping) or source_file.get("sha256") != row.get(
            "sourceSha256"
        ):
            raise ValueError("video playback source frame is not hash-bound")
        previous = counts
    if (
        set(previous.values()) != {0}
        or rows[-1].get("observerHandCardIds") != []
        or len(played_card_ids) != 32
        or len(set(played_card_ids)) != 32
        or set(observer_played_ids) != set(rows[0]["observerHandCardIds"])
    ):
        raise ValueError("video playback evidence does not finish with empty hands")


def _verified_manifest_json(
    manifest: Mapping[str, Any],
    batch_directory: Path,
    key: str,
) -> dict[str, Any]:
    reference = dict(dict(manifest.get("files") or {}).get(key) or {})
    name = str(reference.get("name") or "")
    if not name or Path(name).name != name:
        raise ValueError(f"video {key} artifact name is invalid")
    path = batch_directory / name
    if (
        not path.is_file()
        or path.stat().st_size != reference.get("bytes")
        or _sha256(path) != reference.get("sha256")
    ):
        raise ValueError(f"video {key} artifact hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"video {key} artifact must be a JSON object")
    return value


def _pair(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        raise ValueError(f"video lifecycle {label} must be a two-team integer pair")
    return list(value)


def _validate_lifecycle(
    manifest: Mapping[str, Any],
    batch_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _verified_manifest_json(manifest, batch_directory, "scene:continuation")
    public = next(
        (
            dict(layer.get("content") or {})
            for layer in context.get("layers") or ()
            if layer.get("id") == "public-lifecycle"
            and layer.get("disclosure") == "post_reveal_analysis"
        ),
        None,
    )
    if public is None:
        raise ValueError("video continuation lacks its post-reveal lifecycle")
    plays = public.get("plays")
    if not isinstance(plays, list) or len(plays) != 32:
        raise ValueError("video lifecycle must contain the complete 32-play trace")
    projects = public.get("projects")
    if not isinstance(projects, list):
        raise ValueError("video lifecycle projects must be a list")
    for project in projects:
        if (
            not isinstance(project, Mapping)
            or type(project.get("revealed")) is not bool
            or type(project.get("scoring")) is not bool
        ):
            raise ValueError("video project lifecycle is incomplete")
        cards = project.get("cards")
        if project["scoring"]:
            if (
                project["revealed"] is not True
                or not isinstance(cards, list)
                or not cards
            ):
                raise ValueError("video project lifecycle is incomplete")
        elif project["revealed"] is not False or cards not in (None, []):
            raise ValueError("video non-scoring project lifecycle is not observer-safe")
    round_score = public.get("roundScore")
    if not isinstance(round_score, Mapping):
        raise ValueError("flagship video lifecycle lacks terminal score evidence")
    score_pairs = {
        key: _pair(round_score.get(key), key)
        for key in (
            "cardGamePoints",
            "projectPoints",
            "balootPoints",
            "awardedPoints",
        )
    }
    if any(score_pairs["projectPoints"]) and not projects:
        raise ValueError("video score has project points without project evidence")
    pre_score = _pair(public.get("preDecisionMatchScore"), "preDecisionMatchScore")
    final_score = _pair(public.get("matchScore"), "matchScore")
    if final_score != [
        pre_score[index] + score_pairs["awardedPoints"][index] for index in range(2)
    ]:
        raise ValueError(
            "video match score does not transition by awarded round points"
        )
    evidence = {
        "projectCount": len(projects),
        "projectPoints": score_pairs["projectPoints"],
        "preDecisionMatchScore": pre_score,
        "awardedPoints": score_pairs["awardedPoints"],
        "finalMatchScore": final_score,
        "roundWinner": round_score.get("winner"),
    }
    return public, {**evidence, "sha256": canonical_sha256(evidence)}


def _semantic_variant_profile(
    result: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = list(dict(result.get("predicateEvaluation") or {}).get("evidence") or ())

    def observed_leaf(item: Mapping[str, Any]) -> dict[str, Any]:
        observed = dict(item.get("observed") or {})
        while isinstance(observed.get("observed"), Mapping):
            observed = dict(observed["observed"])
        return observed

    winning_actions = [
        action
        for item in evidence
        if item.get("kind") == "proof.winning_root_action_matches"
        for action in observed_leaf(item).get("winningActions") or ()
        if action.get("winning") is True
    ]
    action = dict(winning_actions[0]) if len(winning_actions) == 1 else {}
    card = dict(action.get("card") or {})
    actor = str(action.get("actor") or "")
    low_winners = sorted(
        (
            str(dict(observed.get("winner") or {}).get("actionId") or ""),
            str(
                dict(dict(observed.get("winner") or {}).get("card") or {}).get("rank")
                or ""
            ),
        )
        for item in evidence
        if item.get("kind") == "play.trick_winner_matches"
        for observed in (dict(item.get("observed") or {}),)
        if dict(dict(observed.get("winner") or {}).get("card") or {}).get("rank")
        in {"seven", "eight", "nine"}
    )
    seat_ring = ("north", "west", "south", "east")
    pause = lifecycle.get("pauseBeforePlayIndex")
    plays = lifecycle.get("plays")
    if (
        type(pause) is not int
        or not isinstance(plays, list)
        or not 0 <= pause < len(plays)
    ):
        raise ValueError("video diversity lacks a valid decision prelude")
    if actor not in seat_ring:
        actor = str(dict(plays[pause]).get("seat") or "")
    actor_index = seat_ring.index(actor) if actor in seat_ring else 0
    prelude_pattern = [
        {
            "actorOffset": (seat_ring.index(str(play.get("seat"))) - actor_index) % 4,
            "rank": str(dict(play.get("card") or {}).get("rank") or ""),
        }
        for play in plays[:pause]
    ]
    projects = [
        {
            "ownerOffset": (seat_ring.index(str(project.get("owner"))) - actor_index)
            % 4,
            "kind": str(project.get("kind") or ""),
            "value": project.get("value"),
            "cardRanks": sorted(
                str(dict(project_card).get("rank") or "")
                for project_card in project.get("cards") or ()
            ),
        }
        for project in lifecycle.get("projects") or ()
    ]
    principal_variation_ranks = [
        Card.parse(action_id).rank.name.lower()
        for action_id in dict(result.get("proofOutcome") or {}).get(
            "principalVariation"
        )
        or ()
    ]
    profile = {
        "decisionKind": dict(result.get("scenarioProjection") or {}).get(
            "decisionKind"
        ),
        "actorSeat": actor,
        "winningAction": dict(result.get("proofOutcome") or {}).get("selectedAction"),
        "rootCardRank": card.get("rank"),
        "rootCardSuit": card.get("suit"),
        "rootCardCode": card.get("code"),
        "lowWinningRanks": [rank for _action_id, rank in low_winners],
        "projectConstruction": projects,
        "principalVariationRanks": principal_variation_ranks,
        # Seat-relative ranks deliberately ignore cosmetic suit/seat renaming.
        "decisionPreludePattern": prelude_pattern,
    }
    return {**profile, "sha256": canonical_sha256(profile)}


def _validate_semantic_diversity(
    profiles_by_code: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        len(profiles_by_code) != 5
        or len({profile.get("sha256") for profile in profiles_by_code.values()}) != 5
    ):
        raise ValueError("five-video variants are not semantically distinct")
    dimensions = {
        "actorSeat": len(
            {str(profile.get("actorSeat")) for profile in profiles_by_code.values()}
        ),
        "rootCardSuit": len(
            {str(profile.get("rootCardSuit")) for profile in profiles_by_code.values()}
        ),
        "winningAction": len(
            {str(profile.get("winningAction")) for profile in profiles_by_code.values()}
        ),
        "lowWinningRanks": len(
            {
                canonical_sha256(profile.get("lowWinningRanks"))
                for profile in profiles_by_code.values()
            }
        ),
        "projectConstruction": len(
            {
                canonical_sha256(profile.get("projectConstruction"))
                for profile in profiles_by_code.values()
            }
        ),
        "principalVariationRanks": len(
            {
                canonical_sha256(profile.get("principalVariationRanks"))
                for profile in profiles_by_code.values()
            }
        ),
        "decisionPreludePattern": len(
            {
                canonical_sha256(profile.get("decisionPreludePattern"))
                for profile in profiles_by_code.values()
            }
        ),
    }
    meaningful = [
        key
        for key in (
            "actorSeat",
            "rootCardSuit",
            "winningAction",
            "lowWinningRanks",
            "projectConstruction",
            "principalVariationRanks",
            "decisionPreludePattern",
        )
        if dimensions[key] > 1
    ]
    if len(meaningful) < 2 or not any(
        dimensions[key] > 1
        for key in (
            "lowWinningRanks",
            "projectConstruction",
            "principalVariationRanks",
            "decisionPreludePattern",
        )
    ):
        raise ValueError("five-video variants vary only by cosmetic identities")
    core = {
        "variantCount": 5,
        "dimensions": dimensions,
        "varyingDimensions": meaningful,
        "profiles": {
            code: dict(profile) for code, profile in sorted(profiles_by_code.items())
        },
    }
    return {**core, "sha256": canonical_sha256(core)}


def _verified_receipt_artifact(
    reference: Mapping[str, Any],
    repository_root: Path,
) -> Path:
    path = (repository_root / str(reference.get("path") or "")).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError("five-video evidence references an external artifact") from exc
    if (
        not path.is_file()
        or path.stat().st_size != reference.get("bytes")
        or _sha256(path) != reference.get("sha256")
    ):
        raise ValueError("five-video evidence artifact hash mismatch")
    return path


def _required_playing_quality_hash(source: Mapping[str, Any]) -> str:
    policy = dict(source.get("publicationPolicy") or {})
    policy_core = {key: value for key, value in policy.items() if key != "sha256"}
    quality_hash = policy.get("qualityEvidenceHash")
    if (
        source.get("publicationEligible") is not True
        or source.get("auditOnly") is not False
        or source.get("renderPurpose") != "content_candidate"
        or policy.get("sha256") != canonical_sha256(policy_core)
        or policy.get("blockers") != []
        or dict(policy.get("decisionSelection") or {}).get("gatePassed") is not True
        or dict(policy.get("videoProduction") or {}).get("gatePassed") is not True
        or not isinstance(quality_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", quality_hash) is None
    ):
        raise ValueError(
            "five-video intelligent-play evidence is not publication eligible"
        )
    return quality_hash


def validate_existing_evidence(
    receipt_path: Path,
    *,
    repository_root: Path = ROOT,
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "passed" or receipt.get("sha256") != canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    ):
        raise ValueError("five-video evidence receipt identity is invalid")
    result_rows = list(receipt.get("results") or ())
    video_rows = list(receipt.get("videos") or ())
    if len(result_rows) != 5 or len(video_rows) != 5:
        raise ValueError("five-video evidence does not contain five complete variants")
    recipe_document = json.loads(
        _verified_receipt_artifact(
            dict(receipt.get("recipeArtifact") or {}), repository_root
        ).read_text(encoding="utf-8")
    )
    recipe = ResearchRecipeV4.build(recipe_document)
    if recipe.code != receipt.get("recipeCode") or recipe.sha256 != receipt.get(
        "recipeHash"
    ):
        raise ValueError("five-video recipe authority identity is invalid")
    quality_required = recipe.playing_quality is not None

    documents: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        document = json.loads(
            _verified_receipt_artifact(
                dict(row.get("artifact") or {}), repository_root
            ).read_text(encoding="utf-8")
        )
        code = str(row.get("resultCode") or "")
        authority = dict(document.get("researchAuthority") or {})
        qualification = dict(document.get("qualification") or {})
        projection = dict(document.get("scenarioProjection") or {})
        proof = dict(document.get("proofOutcome") or {})
        if (
            document.get("resultCode") != code
            or document.get("disposition") != "qualified"
            or dict(document.get("runtimeCompatibility") or {}).get("status") != "passed"
            or proof.get("completionStatus")
            not in {"exact_complete", "bounded_complete", "policy_complete"}
            or proof.get("completionStatus") != row.get("proofCompletionStatus")
            or proof.get("achievedMode") != row.get("proofAchievedMode")
            or proof.get("claimSatisfied") is not row.get("proofClaimSatisfied")
            or document.get("resultHash") != row.get("resultHash")
            or authority.get("backendId") != "ortools_cp_sat_v3"
            or authority.get("qualificationAuthority") != "cp_sat"
            or authority.get("traceHash") != row.get("traceHash")
            or authority.get("proofOutcomeHash") != row.get("proofArtifactHash")
            or projection.get("exactStateId") != row.get("exactStateId")
            or projection.get("publicProjectionHash") != row.get("publicProjectionHash")
            or proof.get("claimSatisfied") is not True
            or (
                quality_required
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("qualityEvidenceHash") or "")
                )
                is None
            )
            or (
                quality_required
                and (
                    qualification.get("allDecisionPolicyAuditComplete") is not True
                    or qualification.get("playingQualityEvidenceHash")
                    != row.get("qualityEvidenceHash")
                )
            )
        ):
            raise ValueError("five-video result authority identity is invalid")
        documents[code] = document
        identities[code] = {
            "resultHash": row["resultHash"],
            "exactStateId": row["exactStateId"],
            "proofArtifactHash": row["proofArtifactHash"],
            "publicProjectionHash": row["publicProjectionHash"],
            "proofCompletionStatus": proof.get("completionStatus"),
            "proofAchievedMode": proof.get("achievedMode"),
            "proofClaimSatisfied": proof.get("claimSatisfied"),
            "qualityEvidenceHash": row.get("qualityEvidenceHash"),
        }

    audited_rows: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    for row in video_rows:
        manifest_path = _verified_receipt_artifact(
            dict(row.get("manifestArtifact") or {}), repository_root
        )
        _verified_receipt_artifact(
            dict(row.get("videoArtifact") or {}), repository_root
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("manifestSha256") != canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifestSha256"}
        ):
            raise ValueError("five-video manifest identity is invalid")
        source = dict(manifest.get("source") or {})
        code = str(source.get("code") or "")
        identity = identities.get(code)
        quality_hash = (
            _required_playing_quality_hash(source) if quality_required else None
        )
        if (
            identity is None
            or source.get("sourceSchemaVersion") != 3
            or source.get("publicProjectionHash") != identity["publicProjectionHash"]
            or source.get("proofReference") != code
            or source.get("proofCompletionStatus") != identity["proofCompletionStatus"]
            or source.get("proofAchievedMode") != identity["proofAchievedMode"]
            or source.get("proofClaimSatisfied") is not identity["proofClaimSatisfied"]
            or manifest.get("observerSafe") is not True
            or (
                quality_required and quality_hash != identity.get("qualityEvidenceHash")
            )
        ):
            raise ValueError("five-video observer-safe source identity is invalid")
        _validate_playback(manifest)
        lifecycle, lifecycle_evidence = _validate_lifecycle(
            manifest, manifest_path.parent
        )
        profile = _semantic_variant_profile(documents[code], lifecycle)
        profiles[code] = profile
        audited_rows.append(
            {
                "resultCode": code,
                **identity,
                "playbackBoundaryCount": len(manifest["playbackEvidence"]),
                "visibleCardStateVerified": False,
                "lifecycleEvidence": lifecycle_evidence,
                "semanticVariantProfileHash": profile["sha256"],
            }
        )
    if {row["resultCode"] for row in audited_rows} != set(documents):
        raise ValueError("five-video manifests do not own the retained results")
    semantic_diversity = _validate_semantic_diversity(profiles)
    decision_kinds = sorted(
        {
            str(dict(document.get("scenarioProjection") or {}).get("decisionKind"))
            for document in documents.values()
        }
    )
    if not set(decision_kinds) <= {"bid", "escalation", "project", "play"}:
        raise ValueError("five-video result decision kind is invalid")
    core = {
        "schemaVersion": 1,
        "kind": "ortools_v3_five_video_semantic_audit",
        "status": "passed",
        "validationClass": (
            "current_intelligent" if quality_required else "historical_integrity"
        ),
        "publicationEligible": quality_required,
        "sourceEvidence": _artifact_reference(receipt_path, repository_root),
        "decisionKindCoverage": {
            "observed": decision_kinds,
            "scope": "five_flagship_variants",
        },
        "semanticDiversity": semantic_diversity,
        "results": sorted(audited_rows, key=lambda item: item["resultCode"]),
    }
    return {**core, "sha256": canonical_sha256(core)}


def build_receipt(
    *,
    source_identity: Mapping[str, Any],
    recipe: ResearchRecipeV4,
    recipe_artifact: Mapping[str, Any],
    run: Mapping[str, Any],
    prepared: Mapping[str, Any],
    results: list[Mapping[str, Any]],
    result_artifacts: Mapping[str, Mapping[str, Any]],
    batch: Mapping[str, Any],
    execution: Mapping[str, Any],
    batch_directory: Path,
    repository_root: Path = ROOT,
    research_wall_seconds: float,
    render_wall_seconds: float,
) -> dict[str, Any]:
    if batch.get("sha256") != canonical_sha256(
        {key: value for key, value in batch.items() if key != "sha256"}
    ):
        raise ValueError("render batch receipt is invalid")
    if execution.get("sha256") != canonical_sha256(
        {key: value for key, value in execution.items() if key != "sha256"}
    ):
        raise ValueError("render batch execution receipt is invalid")
    if len(results) != 5 or len(prepared.get("selectedResultCodes") or ()) != 5:
        raise ValueError("five-video receipt requires exactly five selected results")
    if execution.get("status") != "completed" or len(execution.get("items") or ()) != 5:
        raise ValueError("five-video render batch is incomplete")
    if prepared.get("sha256") != canonical_sha256(
        {key: value for key, value in prepared.items() if key != "sha256"}
    ):
        raise ValueError("selected-run preparation receipt is invalid")
    result_rows: list[dict[str, Any]] = []
    prepared_by_code = {
        str(item["source"]["code"]): dict(item) for item in prepared["items"]
    }
    quality_required = getattr(recipe, "playing_quality", None) is not None
    for result in results:
        authority = dict(result.get("researchAuthority") or {})
        qualification = dict(result.get("qualification") or {})
        projection = dict(result.get("scenarioProjection") or {})
        proof = dict(result.get("proofOutcome") or {})
        compatibility = dict(result.get("runtimeCompatibility") or {})
        prepared_item = prepared_by_code.get(str(result.get("resultCode") or ""))
        if not isinstance(prepared_item, Mapping):
            raise ValueError("prepared video authority receipt is missing")
        authority_receipt = dict(prepared_item["authorityReceipt"])
        prepared_source = dict(prepared_item.get("source") or {})
        quality_hash = (
            _required_playing_quality_hash(prepared_source)
            if quality_required
            else None
        )
        if (
            result.get("disposition") != "qualified"
            or authority.get("backendId") != "ortools_cp_sat_v3"
            or authority.get("qualificationAuthority") != "cp_sat"
            or compatibility.get("status") != "passed"
            or proof.get("claimSatisfied") is not True
            or proof.get("completionStatus")
            not in {"exact_complete", "bounded_complete", "policy_complete"}
            or projection.get("exactStateId") != authority_receipt.get("exactStateId")
            or projection.get("publicProjectionHash")
            != dict(prepared_item["source"]).get("publicProjectionHash")
            or authority.get("proofOutcomeHash")
            != authority_receipt.get("proofArtifactSha256")
            or (
                quality_required
                and (
                    qualification.get("allDecisionPolicyAuditComplete") is not True
                    or qualification.get("playingQualityEvidenceHash")
                    != quality_hash
                )
            )
        ):
            raise ValueError(
                "five-video result is not qualified by the OR-Tools V4 authority"
            )
        result_rows.append(
            {
                "resultCode": result["resultCode"],
                "resultHash": result["resultHash"],
                "exactStateId": authority_receipt["exactStateId"],
                "stateHash": authority_receipt["stateHash"],
                "exactStateHash": authority_receipt["exactStateHash"],
                "publicProjectionHash": projection["publicProjectionHash"],
                "traceHash": authority["traceHash"],
                "proofOutcomeHash": authority["proofOutcomeHash"],
                "proofArtifactHash": authority_receipt["proofArtifactSha256"],
                "proofCompletionStatus": proof["completionStatus"],
                "proofAchievedMode": proof["achievedMode"],
                "proofClaimSatisfied": proof["claimSatisfied"],
                "renderBundleHash": prepared_item["bundleArtifact"]["ownerHash"],
                "configurationBundleHash": authority_receipt["configurationBundleHash"],
                "researchAuthorityHash": authority["sha256"],
                "executionReceiptHash": result["executionReceipt"]["sha256"],
                "stageTimings": dict(
                    result["executionReceipt"].get("stageTimings") or {}
                ),
                **(
                    {"qualityEvidenceHash": quality_hash}
                    if quality_hash is not None
                    else {}
                ),
                "artifact": dict(result_artifacts[result["resultCode"]]),
            }
        )
    for field in (
        "resultCode",
        "resultHash",
        "exactStateId",
        "publicProjectionHash",
        "traceHash",
    ):
        if len({row[field] for row in result_rows}) != 5:
            raise ValueError(f"five-video variants are not distinct by {field}")
    result_by_code = {row["resultCode"]: row for row in result_rows}

    video_rows: list[dict[str, Any]] = []
    semantic_profiles: dict[str, dict[str, Any]] = {}
    for item in execution["items"]:
        manifest_path = batch_directory / str(item["manifestName"])
        if _sha256(manifest_path) != item["manifestSha256"]:
            raise ValueError("render batch manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_core = {
            key: value for key, value in manifest.items() if key != "manifestSha256"
        }
        if manifest.get("manifestSha256") != canonical_sha256(manifest_core):
            raise ValueError("rendered video manifest identity mismatch")
        for artifact in manifest.get("files", {}).values():
            path = batch_directory / str(artifact["name"])
            if (
                not path.is_file()
                or path.stat().st_size != artifact["bytes"]
                or _sha256(path) != artifact["sha256"]
            ):
                raise ValueError("rendered video artifact hash mismatch")
        _validate_playback(manifest)
        source = dict(manifest.get("source") or {})
        result_identity = result_by_code.get(str(source.get("code") or ""))
        quality_hash = (
            _required_playing_quality_hash(source) if quality_required else None
        )
        if (
            result_identity is None
            or source.get("sourceSchemaVersion") != 3
            or source.get("publicProjectionHash")
            != result_identity["publicProjectionHash"]
            or source.get("proofReference") != result_identity["resultCode"]
            or source.get("proofCompletionStatus")
            != result_identity["proofCompletionStatus"]
            or source.get("proofAchievedMode") != result_identity["proofAchievedMode"]
            or source.get("proofClaimSatisfied")
            is not result_identity["proofClaimSatisfied"]
            or (
                quality_required
                and quality_hash != result_identity.get("qualityEvidenceHash")
            )
        ):
            raise ValueError("rendered video lost its V3 result/proof identity")
        lifecycle, lifecycle_evidence = _validate_lifecycle(
            manifest,
            batch_directory,
        )
        result_document = next(
            result
            for result in results
            if result.get("resultCode") == result_identity["resultCode"]
        )
        semantic_profile = _semantic_variant_profile(result_document, lifecycle)
        semantic_profiles[result_identity["resultCode"]] = semantic_profile
        video_path = batch_directory / str(manifest["files"]["video"]["name"])
        if _sha256(video_path) != manifest["files"]["video"]["sha256"]:
            raise ValueError("rendered video hash mismatch")
        if (
            manifest.get("observerSafe") is not True
            or manifest.get("deterministic") is not True
        ):
            raise ValueError("rendered video lost observer-safe deterministic status")
        video_rows.append(
            {
                "ordinal": item["ordinal"],
                "resultCode": manifest["source"]["code"],
                "bundleHash": item["bundleSha256"],
                "manifestHash": item["manifestSha256"],
                "videoHash": manifest["files"]["video"]["sha256"],
                "mediaProbeHash": _canonical_document_hash(
                    dict(manifest["mediaProbe"])
                ),
                "playbackEvidenceHash": canonical_sha256(manifest["playbackEvidence"]),
                "playbackBoundaryCount": len(manifest["playbackEvidence"]),
                "visibleCardStateVerified": False,
                "exactStateId": result_identity["exactStateId"],
                "proofArtifactHash": result_identity["proofArtifactHash"],
                "lifecycleEvidence": lifecycle_evidence,
                "semanticVariantProfileHash": semantic_profile["sha256"],
                "manifestArtifact": _artifact_reference(manifest_path, repository_root),
                "videoArtifact": _artifact_reference(video_path, repository_root),
            }
        )
    if {row["resultCode"] for row in video_rows} != set(prepared_by_code):
        raise ValueError("rendered videos do not match the selected Research results")
    semantic_diversity = _validate_semantic_diversity(semantic_profiles)
    variants = [
        {
            "variantId": row["resultCode"],
            "recipe": dict(recipe_artifact),
            "result": dict(result_by_code[row["resultCode"]]["artifact"]),
            "videoManifest": dict(row["manifestArtifact"]),
            "video": dict(row["videoArtifact"]),
        }
        for row in sorted(video_rows, key=lambda item: item["ordinal"])
    ]

    performance = dict(run.get("performanceReceipt") or {})
    core = {
        "schemaVersion": 1,
        "kind": "ortools_v4_five_video_release_evidence",
        "status": "passed",
        "source": {
            "identity": dict(source_identity),
            "identitySha256": canonical_sha256(dict(source_identity)),
        },
        "configuration": dict(recipe.configuration),
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "recipeArtifact": dict(recipe_artifact),
        "runCode": run["runCode"],
        "runHash": run["runHash"],
        "outputReceiptHash": prepared["outputReceiptHash"],
        "diversitySelectionHash": prepared["diversitySelectionHash"],
        "semanticDiversity": semantic_diversity,
        "performanceReceiptHash": _canonical_document_hash(performance),
        "batchHash": batch["sha256"],
        "batchExecutionHash": execution["sha256"],
        "stageTimings": {
            "researchWallSeconds": research_wall_seconds,
            "renderWallSeconds": render_wall_seconds,
            "research": dict(performance.get("stageTimings") or {}),
        },
        "results": result_rows,
        "videos": sorted(video_rows, key=lambda row: row["ordinal"]),
        "variants": variants,
    }
    return {**core, "sha256": canonical_sha256(core)}


def _persist_terminal_research_failure(
    output_root: Path,
    recipe: ResearchRecipeV4,
    run_value: Mapping[str, Any],
    *,
    research_wall_seconds: float,
) -> Path:
    """Preserve the complete terminal run while keeping CLI errors bounded."""

    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / "research-run-terminal.json"
    run_payload = (
        json.dumps(dict(run_value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    atomic_write_text(run_path, run_payload)
    core = {
        "schemaVersion": 1,
        "kind": "ortools_v4_five_video_terminal_research_receipt",
        "status": "incomplete",
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "runCode": run_value.get("runCode"),
        "runHash": run_value.get("runHash"),
        "terminalStatus": run_value.get("status"),
        "scenarioResearchStatus": run_value.get("scenarioResearchStatus"),
        "qualifiedCount": run_value.get("qualifiedCount"),
        "selectedCount": run_value.get("selectedCount"),
        "researchWallSeconds": research_wall_seconds,
        "runArtifact": {
            "name": run_path.name,
            "bytes": len(run_payload.encode("utf-8")),
            "sha256": hashlib.sha256(run_payload.encode("utf-8")).hexdigest(),
        },
    }
    receipt_path = output_root / "five-video-research-incomplete.json"
    atomic_write_text(
        receipt_path,
        json.dumps(
            {**core, "sha256": canonical_sha256(core)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return receipt_path


def run(
    recipe_payload: Mapping[str, Any],
    *,
    output_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    from baloot_platform.service import api

    ensure_operator_bundle(
        api.CONTAINER,
        str(
            dict(recipe_payload.get("configuration") or {}).get("bundleRevisionId")
            or ""
        ),
    )
    intent_hash = canonical_sha256(dict(recipe_payload))
    stored = api.RESEARCH.create_recipe(
        recipe_payload,
        idempotency_key=f"five-video-recipe-{intent_hash[:24]}",
    )
    normalized = dict(stored)
    normalized.pop("lifecycleStatus", None)
    recipe = validate_recipe(normalized)
    source_identity = git_identity(ROOT)
    source_hash = canonical_sha256(source_identity)
    research_started = time.perf_counter()
    submitted = api.RESEARCH.submit(
        stored["code"],
        idempotency_key=f"five-video-run-{recipe.sha256[:16]}-{source_hash[:16]}",
    )
    deadline = time.monotonic() + timeout_seconds
    run_value = submitted
    while run_value["status"] not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            api.RESEARCH.cancel(submitted["runCode"])
            raise TimeoutError(
                "five-video Research campaign exceeded its declared timeout"
            )
        time.sleep(0.1)
        run_value = api.RESEARCH.get_run(submitted["runCode"])
    research_wall = time.perf_counter() - research_started
    if (
        run_value["status"] != "completed"
        or run_value.get("qualifiedCount") != 5
        or run_value.get("selectedCount") != 5
    ):
        failure_receipt = _persist_terminal_research_failure(
            output_root,
            recipe,
            run_value,
            research_wall_seconds=research_wall,
        )
        raise RuntimeError(
            "five-video Research campaign did not qualify five results; "
            f"terminal evidence: {failure_receipt}"
        )
    prepared = api.RESEARCH_RENDERS.prepare_selected_run(
        run_value["runCode"],
        export_preset="hooked_flagship",
    )
    bundles: dict[str, RenderBundleV1] = {}
    requests: list[tuple[RenderBundleV1, str]] = []
    proof_hashes: dict[str, str] = {}
    for item in prepared["items"]:
        bundle = _load_bundle(
            api.RESEARCH_RENDERS.artifact_store,
            item["bundleArtifact"],
        )
        bundles[bundle.bundle_sha256] = bundle
        requests.append((bundle, str(item["basename"])))
        proof_hashes[bundle.bundle_sha256] = str(item["proofArtifact"]["sha256"])
    batch = RenderBatchV1.create(requests, proof_artifact_hashes=proof_hashes)
    render_started = time.perf_counter()
    execution = api.RENDER_BATCHES.batch_executor.execute_batch(
        batch,
        bundles=bundles,
        output_root=output_root,
        playback_evidence=True,
    )
    render_wall = time.perf_counter() - render_started
    results = [
        api.RESEARCH.get_result(code) for code in prepared["selectedResultCodes"]
    ]
    if any(result is None for result in results):
        raise RuntimeError(
            "selected Research result disappeared before evidence assembly"
        )
    recipe_artifact, result_artifacts = persist_research_documents(
        output_root,
        recipe,
        results,
    )
    receipt = build_receipt(
        source_identity=source_identity,
        recipe=recipe,
        recipe_artifact=recipe_artifact,
        run=run_value,
        prepared=prepared,
        results=results,
        result_artifacts=result_artifacts,
        batch=batch.to_dict(),
        execution=execution.to_dict(),
        batch_directory=output_root / f"batch-{batch.sha256[:16]}",
        research_wall_seconds=research_wall,
        render_wall_seconds=render_wall,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "five-video-evidence.json"
    atomic_write_text(
        destination,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", nargs="?", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/ortools-v4-five-video-evidence",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--validate-existing", type=Path)
    args = parser.parse_args()
    if args.validate_existing is not None:
        if args.recipe is not None:
            parser.error("recipe and --validate-existing are mutually exclusive")
        audit = validate_existing_evidence(args.validate_existing.resolve())
        destination = args.validate_existing.resolve().with_name(
            "five-video-semantic-audit.json"
        )
        atomic_write_text(
            destination,
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(audit, ensure_ascii=True, separators=(",", ":")))
        return
    if args.recipe is None:
        parser.error("recipe is required unless --validate-existing is used")
    with tempfile.TemporaryDirectory(
        prefix="balootx-five-video-evidence-", ignore_cleanup_errors=True
    ) as data_dir:
        os.environ["BALOOT_DATA_DIR"] = data_dir
        os.environ.pop("BALOOT_DATABASE_URL", None)
        receipt = run(
            json.loads(args.recipe.read_text(encoding="utf-8")),
            output_root=args.output_root.resolve(),
            timeout_seconds=args.timeout_seconds,
        )
    print(json.dumps(receipt, ensure_ascii=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

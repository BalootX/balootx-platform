from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.generate_ortools_v4_five_video_evidence import (
    OPERATOR_BUNDLE,
    _semantic_variant_profile,
    _validate_playback,
    _validate_lifecycle,
    _validate_semantic_diversity,
    build_receipt,
    ensure_operator_bundle,
    persist_research_documents,
    run,
    validate_existing_evidence,
    validate_recipe,
)

from baloot_platform.research.contracts import (
    _RESEARCH_IDENTITY_SOURCE_FAMILIES,
    ResearchConfigurationAuthorityV1,
    canonical_sha256,
)


def test_standalone_evidence_bootstraps_the_exact_operator_bundle(
    tmp_path: Path,
) -> None:
    from baloot_platform.configuration import (
        SQLiteConfigurationStore,
        VersionedConfigurationService,
    )

    profile = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs/configuration/control-plane.yaml"
        ).read_text(encoding="utf-8")
    )
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(tmp_path / "configuration.sqlite3"),
        root=Path(__file__).parents[1],
        profile=profile,
    )
    revision_id = ensure_operator_bundle(SimpleNamespace(configuration=service), None)
    resolved = service.resolve(revision_id=revision_id)
    assert resolved["bundle"]["revisionId"] == revision_id
    expected = yaml.safe_load((Path(__file__).parents[1] / OPERATOR_BUNDLE).read_text(encoding="utf-8"))
    assert resolved["bundle"]["version"] == expected["profile"]["version"]
    service.store.close()


def test_run_resolves_operator_bundle_selector_before_strict_recipe_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from baloot_platform.service import api
    from scripts import generate_ortools_v4_five_video_evidence as generator

    calls: list[str] = []
    normalized = {
        "schemaVersion": 4,
        "code": "RCP-TEST",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": "CBR-TEST",
            "semanticBundleHash": "a" * 64,
            "effectiveRuntimeConfigHash": "b" * 64,
            "identities": {},
        },
    }

    class FakeResearch:
        def create_recipe(
            self, payload: object, *, idempotency_key: str
        ) -> dict[str, object]:
            calls.append("create")
            assert payload == {
                "schemaVersion": 4,
                "configuration": {"schemaVersion": 1, "bundleRevisionId": "CBR-TEST"},
            }
            assert idempotency_key.startswith("five-video-recipe-")
            return {**normalized, "lifecycleStatus": "published"}

        def submit(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            calls.append("submit")
            raise RuntimeError("stop after ordering check")

    recipe = SimpleNamespace(code="RCP-TEST", sha256="c" * 64)

    def fake_validate(value: object) -> object:
        calls.append("validate")
        assert value == normalized
        return recipe

    monkeypatch.setattr(api, "RESEARCH", FakeResearch())
    monkeypatch.setattr(generator, "ensure_operator_bundle", lambda *_args: None)
    monkeypatch.setattr(generator, "validate_recipe", fake_validate)
    monkeypatch.setattr(generator, "git_identity", lambda _root: {"revision": "d" * 40})

    with pytest.raises(RuntimeError, match="ordering check"):
        run(
            {
                "schemaVersion": 4,
                "configuration": {"schemaVersion": 1, "bundleRevisionId": "CBR-TEST"},
            },
            output_root=tmp_path,
            timeout_seconds=1.0,
        )
    assert calls == ["create", "validate", "submit"]


def test_run_preserves_terminal_campaign_without_embedding_full_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from baloot_platform.service import api
    from scripts import generate_ortools_v4_five_video_evidence as generator

    terminal = {
        "runCode": "RUN-TERMINAL",
        "runHash": "a" * 64,
        "status": "completed",
        "scenarioResearchStatus": "proof_incomplete",
        "qualifiedCount": 0,
        "selectedCount": 0,
        "largeEvidence": "x" * 100_000,
    }

    class FakeResearch:
        def create_recipe(
            self, payload: object, *, idempotency_key: str
        ) -> dict[str, object]:
            return {"code": "RCP-TEST", "lifecycleStatus": "published"}

        def submit(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return terminal

    recipe = SimpleNamespace(code="RCP-TEST", sha256="b" * 64)
    monkeypatch.setattr(api, "RESEARCH", FakeResearch())
    monkeypatch.setattr(generator, "ensure_operator_bundle", lambda *_args: None)
    monkeypatch.setattr(generator, "validate_recipe", lambda _value: recipe)
    monkeypatch.setattr(generator, "git_identity", lambda _root: {"revision": "c" * 40})

    with pytest.raises(RuntimeError) as raised:
        run({}, output_root=tmp_path, timeout_seconds=1.0)

    assert len(str(raised.value)) < 500
    receipt = json.loads(
        (tmp_path / "five-video-research-incomplete.json").read_text(encoding="utf-8")
    )
    retained = json.loads(
        (tmp_path / "research-run-terminal.json").read_text(encoding="utf-8")
    )
    assert receipt["scenarioResearchStatus"] == "proof_incomplete"
    assert (
        receipt["runArtifact"]["bytes"]
        == (tmp_path / "research-run-terminal.json").stat().st_size
    )
    assert retained == terminal


def _playback_rows() -> list[dict[str, object]]:
    seats = ("north", "west", "south", "east")
    observer_cards = list(range(3, 32, 4))
    counts = {seat: 8 for seat in seats}
    rows: list[dict[str, object]] = []
    for count in range(33):
        if count:
            actor = seats[(count - 1) % 4]
            counts[actor] -= 1
            played = {"seat": actor, "card": {"id": count - 1}}
        else:
            played = None
        position = count % 4
        current_cards = list(range(count - position, count)) if position else []
        rows.append(
            {
                "completedPlayCount": count,
                "frame": count,
                "source": f"playback-{count:02d}.png",
                "playedCard": played,
                "observerHandCardIds": [
                    card_id for card_id in observer_cards if card_id >= count
                ],
                "handCounts": dict(counts),
                "currentTrickCardIds": current_cards,
                "tableCardIds": (
                    list(range(count - 4, count))
                    if count and not position
                    else current_cards
                ),
                "completedTrickIndex": count // 4 - 1
                if count and not position
                else None,
                "sourceSha256": f"{count:064x}",
                "encodedFrameComparison": {
                    "frame": count,
                    "meanAbsoluteError": 0.0,
                    "psnrDb": 999.0,
                },
            }
        )
    rows[-1]["observerHandCardIds"] = []
    return rows


def _flagship_predicate_evidence(index: int) -> list[dict[str, object]]:
    tail_ranks = ("seven", "eight", "nine")
    return [
        {
            "kind": "proof.winning_root_action_matches",
            "observed": {
                "winningActions": [
                    {
                        "winning": True,
                        "actor": "west",
                        "card": {
                            "rank": "ace",
                            "suit": "diamonds",
                            "code": "DA",
                        },
                    }
                ]
            },
        },
        *[
            {
                "kind": "play.trick_winner_matches",
                "observed": {
                    "winner": {
                        "actionId": f"WINNER-{trick}",
                        "card": {"rank": rank},
                    }
                },
            }
            for trick, rank in (
                (6, "eight"),
                (7, tail_ranks[index % len(tail_ranks)]),
                (8, tail_ranks[(index + 1) % len(tail_ranks)]),
            )
        ],
    ]


def _flagship_lifecycle(index: int) -> dict[str, object]:
    seats = ("north", "west", "south", "east")
    ranks = ("seven", "eight", "nine", "ten", "jack", "queen", "king", "ace")
    plays = [
        {
            "seat": seats[play_index % 4],
            "card": {
                "rank": ranks[(play_index + (index if play_index == 0 else 0)) % 8]
            },
        }
        for play_index in range(32)
    ]
    project_start = 0 if index < 3 else 1
    project_cards = [{"rank": ranks[project_start + offset]} for offset in range(5)]
    return {
        "plays": plays,
        "pauseBeforePlayIndex": 24,
        "projects": [
            {
                "owner": "north",
                "kind": "hundred_sequence",
                "value": 20,
                "revealed": True,
                "scoring": True,
                "cards": project_cards,
            }
        ],
        "preDecisionMatchScore": [3, 7],
        "matchScore": [3, 53],
        "roundScore": {
            "cardGamePoints": [0, 26],
            "projectPoints": [20, 0],
            "balootPoints": [0, 0],
            "awardedPoints": [0, 46],
            "winner": "west_east",
        },
    }


def test_playback_receipt_requires_every_card_and_trick_boundary() -> None:
    rows = _playback_rows()
    files = {
        f"playback:{count:02d}": {"sha256": row["sourceSha256"]}
        for count, row in enumerate(rows)
    }
    manifest = {
        "source": {"observerSeat": "east"},
        "playbackEvidence": rows,
        "files": files,
    }
    _validate_playback(manifest)
    outcome = {**rows[-1], "checkpointPhase": "outcome", "tableCardIds": [],
               "frame": 4200, "encodedFrameComparison": {"frame": 4200}}
    files["playback:outcome"] = {"sha256": outcome["sourceSha256"]}
    rows.append(outcome)
    _validate_playback(manifest)
    outcome["tableCardIds"] = [1]
    with pytest.raises(ValueError, match="outcome checkpoint"):
        _validate_playback(manifest)
    rows.pop()
    for row in rows:
        row["logicalTableCardIds"] = row.pop("tableCardIds")
        row["visibleCardStateVerified"] = False
    _validate_playback(manifest)
    rows[0]["tableCardIds"] = []
    with pytest.raises(ValueError, match="competing table-state"):
        _validate_playback(manifest)
    del rows[0]["tableCardIds"]
    rows[8]["currentTrickCardIds"] = [1]
    try:
        _validate_playback(manifest)
    except ValueError as exc:
        assert "current trick" in str(exc)
    else:
        raise AssertionError("drifted trick progression must fail")
    rows = _playback_rows()
    rows[4]["observerHandCardIds"] = rows[3]["observerHandCardIds"]
    manifest["playbackEvidence"] = rows
    with pytest.raises(ValueError, match="observer hand"):
        _validate_playback(manifest)


def test_semantic_diversity_rejects_cosmetic_identity_only_variants() -> None:
    profile = {
        "actorSeat": "west",
        "rootCardSuit": "diamonds",
        "lowWinningRanks": ["seven", "nine"],
        "projectConstruction": [{"kind": "hundred_sequence"}],
        "decisionPreludePattern": [{"actorOffset": 1, "rank": "seven"}],
    }
    profile["sha256"] = canonical_sha256(profile)
    with pytest.raises(ValueError, match="semantically distinct"):
        _validate_semantic_diversity(
            {f"LAF-{index:010X}": dict(profile) for index in range(5)}
        )


def test_semantic_profile_reads_nested_native_proof_and_continuation() -> None:
    result = {
        "scenarioProjection": {"decisionKind": "play"},
        "proofOutcome": {
            "selectedAction": "CA",
            "principalVariation": ["CA", "C7", "SK", "S8"],
        },
        "predicateEvaluation": {
            "evidence": [
                {
                    "kind": "proof.winning_root_action_matches",
                    "observed": {
                        "observed": {
                            "winningActions": [
                                {
                                    "winning": True,
                                    "actor": "west",
                                    "card": {
                                        "rank": "ace",
                                        "suit": "clubs",
                                        "code": "CA",
                                    },
                                }
                            ]
                        }
                    },
                }
            ]
        },
    }
    lifecycle = {
        "pauseBeforePlayIndex": 1,
        "plays": [
            {"seat": "north", "card": {"rank": "seven"}},
            {"seat": "west", "card": {"rank": "ace"}},
        ],
        "projects": [],
    }

    profile = _semantic_variant_profile(result, lifecycle)

    assert profile["rootCardRank"] == "ace"
    assert profile["rootCardSuit"] == "clubs"
    assert profile["principalVariationRanks"] == [
        "ace",
        "seven",
        "king",
        "eight",
    ]


def test_semantic_diversity_counts_history_and_proof_continuation() -> None:
    profiles = {}
    for index in range(5):
        profile = {
            "actorSeat": "west",
            "rootCardSuit": "clubs",
            "winningAction": "CA",
            "lowWinningRanks": ["nine", "seven"],
            "projectConstruction": [{"kind": "hundred_sequence"}],
            "principalVariationRanks": [
                "ace",
                ("ten", "queen", "king")[index % 3],
                "seven",
            ],
            "decisionPreludePattern": [
                {"actorOffset": 1, "rank": ("seven", "eight")[index % 2]},
                {"actorOffset": 2, "rank": str(index)},
            ],
        }
        profile["sha256"] = canonical_sha256(profile)
        profiles[f"LAF-{index:010X}"] = profile

    evidence = _validate_semantic_diversity(profiles)

    assert evidence["dimensions"]["principalVariationRanks"] == 3
    assert evidence["dimensions"]["decisionPreludePattern"] == 5


def test_lifecycle_requires_project_and_score_transition(tmp_path: Path) -> None:
    lifecycle = _flagship_lifecycle(0)
    lifecycle["matchScore"] = [3, 52]
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "id": "public-lifecycle",
                        "disclosure": "post_reveal_analysis",
                        "content": lifecycle,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "files": {
            "scene:continuation": {
                "name": context_path.name,
                "bytes": context_path.stat().st_size,
                "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            }
        }
    }
    with pytest.raises(ValueError, match="does not transition"):
        _validate_lifecycle(manifest, tmp_path)


def test_lifecycle_preserves_losing_project_without_revealing_cards(
    tmp_path: Path,
) -> None:
    lifecycle = _flagship_lifecycle(0)
    lifecycle["projects"].append(
        {
            "owner": "west",
            "kind": "siri",
            "value": 4,
            "revealed": False,
            "scoring": False,
            "cards": None,
        }
    )
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "id": "public-lifecycle",
                        "disclosure": "post_reveal_analysis",
                        "content": lifecycle,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "files": {
            "scene:continuation": {
                "name": context_path.name,
                "bytes": context_path.stat().st_size,
                "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
            }
        }
    }
    _validate_lifecycle(manifest, tmp_path)
    lifecycle["projects"][-1]["cards"] = [{"rank": "seven"}]
    context_path.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "id": "public-lifecycle",
                        "disclosure": "post_reveal_analysis",
                        "content": lifecycle,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest["files"]["scene:continuation"].update(
        bytes=context_path.stat().st_size,
        sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="observer-safe"):
        _validate_lifecycle(manifest, tmp_path)


def test_five_video_receipt_binds_source_results_states_proofs_and_media(
    tmp_path: Path,
) -> None:
    recipe_payload = json.loads(
        (
            Path(__file__).parents[1]
            / "examples/research-scenario-v4-flagship-exact-infeasible.json"
        ).read_text(encoding="utf-8")
    )
    # Synthetic receipt binding only; this is not a flagship search qualification.
    recipe_payload.update(code="RCP-RECEIPTTEST",
        output={"kind": "laffa_collection", "resultCount": 5},
        renderPolicy={"mode": "on_demand", "videoCount": 5})
    recipe_payload["scenario"]["constraints"]["children"] = [
        row for row in recipe_payload["scenario"]["constraints"]["children"]
        if row["kind"] in {"auction.contract", "auction.buyer"}
    ]
    recipe_payload["resources"].update(seedCount=5, seedsPerShard=1)
    recipe_payload["diversity"]["topKPerCell"] = 5
    families = sorted(set().union(*_RESEARCH_IDENTITY_SOURCE_FAMILIES.values()))
    components = {
        family: {
            "revisionId": f"CPR-{index:024X}",
            "profileId": f"{family}_test",
            "version": "1.0.0",
            "payloadSha256": f"{index + 1:064x}",
        }
        for index, family in enumerate(families)
    }
    identities = {
        role: {
            "kind": role,
            "id": f"{role}_test",
            "version": "1.0.0",
            "sha256": f"{index + 100:064x}",
            "sourceRevisions": {
                family: components[family]["revisionId"]
                for family in sorted(source_families)
            },
        }
        for index, (role, source_families) in enumerate(
            sorted(_RESEARCH_IDENTITY_SOURCE_FAMILIES.items())
        )
    }
    recipe_payload["configuration"] = ResearchConfigurationAuthorityV1.build(
        {
            "schemaVersion": 1,
            "bundleRevisionId": "CBR-TEST",
            "bundleId": "test_bundle",
            "bundleVersion": "1.0.0",
            "semanticBundleHash": "a" * 64,
            "effectiveConfigHash": "b" * 64,
            "effectiveRuntimeConfigHash": "c" * 64,
            "snapshotContract": "EffectiveRuntimeConfigV2",
            "snapshotArtifact": {
                "uri": f"cas://sha256/{'d' * 64}",
                "sha256": "d" * 64,
                "bytes": 1,
                "mediaType": "application/vnd.balootx.effective-runtime-config-v2+json",
                "name": "effective-runtime-config-v2.json",
            },
            "componentRevisions": components,
            "identities": identities,
        }
    ).to_dict()
    historical_recipe_payload = json.loads(json.dumps(recipe_payload))
    historical_recipe_payload.pop("playingQuality")
    recipe = validate_recipe(historical_recipe_payload)
    result_codes = [f"LAF-{index:010X}" for index in range(5)]
    prepared_core = {
        "schemaVersion": 1,
        "runCode": "RUN-000000000001",
        "runHash": "1" * 64,
        "recipeCode": recipe.code,
        "recipeHash": recipe.sha256,
        "outputReceiptHash": "3" * 64,
        "diversitySelectionHash": "4" * 64,
        "requestedVideoCount": 5,
        "selectedResultCodes": result_codes,
        "items": [
            {
                "source": {
                    "code": code,
                    "publicProjectionHash": f"{index + 40:064x}",
                },
                "bundleArtifact": {
                    "ownerHash": f"{index + 90:064x}",
                    "sha256": f"{index + 95:064x}",
                },
                "authorityReceipt": {
                    "exactStateId": f"STATE-{index:020X}",
                    "stateHash": f"{index + 10:064x}",
                    "exactStateHash": f"{index + 20:064x}",
                    "proofArtifactSha256": f"{index + 60:064x}",
                    "configurationBundleHash": "f" * 64,
                },
            }
            for index, code in enumerate(result_codes)
        ],
    }
    prepared = {**prepared_core, "sha256": canonical_sha256(prepared_core)}
    results = [
        {
            "resultCode": code,
            "resultHash": f"{index + 30:064x}",
            "disposition": "qualified",
            "publicProjectionHash": None,
            "scenarioProjection": {
                "decisionKind": "play",
                "exactStateId": f"STATE-{index:020X}",
                "publicProjectionHash": f"{index + 40:064x}",
            },
            "proofOutcome": {
                "claimSatisfied": True,
                "completionStatus": "exact_complete",
                "achievedMode": "exact",
            },
            "predicateEvaluation": {
                "evidence": _flagship_predicate_evidence(index),
            },
            "runtimeCompatibility": {"status": "passed"},
            "researchAuthority": {
                "backendId": "ortools_cp_sat_v3",
                "qualificationAuthority": "cp_sat",
                "traceHash": f"{index + 50:064x}",
                "proofOutcomeHash": f"{index + 60:064x}",
                "sha256": f"{index + 70:064x}",
            },
            "executionReceipt": {
                "sha256": f"{index + 80:064x}",
                "stageTimings": {"construction": 0.1, "proof": 0.01},
            },
        }
        for index, code in enumerate(result_codes)
    ]
    execution_items = []
    for index in range(5):
        video = tmp_path / f"video-{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        video_hash = hashlib.sha256(video.read_bytes()).hexdigest()
        playback_rows = _playback_rows()
        context = {
            "layers": [
                {
                    "id": "public-lifecycle",
                    "disclosure": "post_reveal_analysis",
                    "content": _flagship_lifecycle(index),
                }
            ]
        }
        context_path = tmp_path / f"video-{index}-scene-context.json"
        context_path.write_text(json.dumps(context), encoding="utf-8")
        manifest = {
            "observerSafe": True,
            "deterministic": True,
            "source": {
                "sourceSchemaVersion": 3,
                "code": result_codes[index],
                "publicProjectionHash": f"{index + 40:064x}",
                "proofReference": result_codes[index],
                "proofCompletionStatus": "exact_complete",
                "proofAchievedMode": "exact",
                "proofClaimSatisfied": True,
                "observerSeat": "east",
            },
            "mediaProbe": {"video": {"decodedFrames": 4320}},
            "playbackEvidence": playback_rows,
            "files": {
                "scene:continuation": {
                    "name": context_path.name,
                    "bytes": context_path.stat().st_size,
                    "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                },
                "video": {
                    "name": video.name,
                    "bytes": video.stat().st_size,
                    "sha256": video_hash,
                },
            },
        }
        for count in range(33):
            source = tmp_path / f"video-{index}-playback-{count:02d}.png"
            source.write_bytes(f"source-{index}-{count}".encode())
            manifest["files"][f"playback:{count:02d}"] = {
                "name": source.name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            playback_rows[count]["sourceSha256"] = manifest["files"][
                f"playback:{count:02d}"
            ]["sha256"]
        manifest["manifestSha256"] = canonical_sha256(manifest)
        manifest_path = tmp_path / f"video-{index}-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        execution_items.append(
            {
                "ordinal": index,
                "bundleSha256": f"{index + 90:064x}",
                "status": "completed",
                "manifestName": manifest_path.name,
                "manifestSha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "errorCode": None,
                "errorDetail": None,
            }
        )
    batch_core = {
        "schemaVersion": 1,
        "items": [{"ordinal": index} for index in range(5)],
    }
    batch = {**batch_core, "sha256": canonical_sha256(batch_core)}
    execution_core = {
        "schemaVersion": 1,
        "batchSha256": batch["sha256"],
        "status": "completed",
        "items": execution_items,
    }
    execution = {**execution_core, "sha256": canonical_sha256(execution_core)}
    run = {
        "runCode": prepared["runCode"],
        "runHash": prepared["runHash"],
        "performanceReceipt": {"stageTimings": {"total": 1.0}},
    }
    recipe_artifact, result_artifacts = persist_research_documents(
        tmp_path / "evidence",
        recipe,
        results,
        repository_root=tmp_path,
    )
    source_identity = {
        "revision": "a" * 40,
        "tree": "b" * 40,
        "scope": "git_visible_product_inputs_v1",
        "excludedPaths": ["outputs/", "release/"],
        "binding": "exact_git_head",
    }

    receipt = build_receipt(
        source_identity=source_identity,
        recipe=recipe,
        recipe_artifact=recipe_artifact,
        run=run,
        prepared=prepared,
        results=results,
        result_artifacts=result_artifacts,
        batch=batch,
        execution=execution,
        batch_directory=tmp_path,
        repository_root=tmp_path,
        research_wall_seconds=1.25,
        render_wall_seconds=2.5,
    )

    assert receipt["status"] == "passed"
    assert len(receipt["results"]) == len(receipt["videos"]) == 5
    assert receipt["recipeArtifact"] == recipe_artifact
    assert receipt["source"]["identity"] == source_identity
    assert {row["artifact"]["path"] for row in receipt["results"]} == {
        f"evidence/results/{code}.json" for code in result_codes
    }
    assert all(row["manifestArtifact"]["path"] for row in receipt["videos"])
    assert all(row["videoArtifact"]["path"] for row in receipt["videos"])
    assert [row["variantId"] for row in receipt["variants"]] == result_codes
    assert all(row["recipe"] == recipe_artifact for row in receipt["variants"])
    assert {row["playbackBoundaryCount"] for row in receipt["videos"]} == {33}
    assert receipt["semanticDiversity"]["variantCount"] == 5
    assert receipt["semanticDiversity"]["dimensions"]["lowWinningRanks"] == 3
    assert receipt["semanticDiversity"]["dimensions"]["projectConstruction"] == 2
    assert receipt["semanticDiversity"]["dimensions"]["decisionPreludePattern"] == 5
    assert all(
        row["lifecycleEvidence"]["finalMatchScore"] == [3, 53]
        for row in receipt["videos"]
    )
    assert all(row["exactStateId"] for row in receipt["videos"])
    assert all(row["proofArtifactHash"] for row in receipt["videos"])
    assert all(row["stateHash"] for row in receipt["results"])
    assert receipt["sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    receipt_path = tmp_path / "five-video-evidence.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    audit = validate_existing_evidence(receipt_path, repository_root=tmp_path)
    assert audit["status"] == "passed"
    assert audit["validationClass"] == "historical_integrity"
    assert audit["publicationEligible"] is False
    assert audit["decisionKindCoverage"] == {
        "observed": ["play"],
        "scope": "five_flagship_variants",
    }
    assert audit["semanticDiversity"] == receipt["semanticDiversity"]

    quality_recipe = validate_recipe(recipe_payload)
    quality_prepared = json.loads(json.dumps(prepared))
    quality_prepared["recipeHash"] = quality_recipe.sha256
    quality_hashes: dict[str, str] = {}
    for index, (prepared_item, execution_item) in enumerate(
        zip(quality_prepared["items"], execution["items"], strict=True)
    ):
        quality_hash = f"{index + 200:064x}"
        quality_hashes[result_codes[index]] = quality_hash
        results[index]["qualification"] = {
            "allDecisionPolicyAuditComplete": True,
            "playingQualityEvidenceHash": quality_hash,
        }
        policy_core = {
            "blockers": [],
            "decisionSelection": {"gatePassed": True},
            "qualityEvidenceHash": quality_hash,
            "videoProduction": {"gatePassed": True},
        }
        publication = {**policy_core, "sha256": canonical_sha256(policy_core)}
        publication_fields = {
            "auditOnly": False,
            "publicationEligible": True,
            "renderPurpose": "content_candidate",
            "publicationPolicy": publication,
        }
        prepared_item["source"].update(publication_fields)
        manifest_path = tmp_path / execution_item["manifestName"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"].update(publication_fields)
        manifest["manifestSha256"] = canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifestSha256"}
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        execution_item["manifestSha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
    quality_prepared["sha256"] = canonical_sha256(
        {key: value for key, value in quality_prepared.items() if key != "sha256"}
    )
    execution["sha256"] = canonical_sha256(
        {key: value for key, value in execution.items() if key != "sha256"}
    )
    quality_recipe_artifact, quality_result_artifacts = persist_research_documents(
        tmp_path / "quality-evidence",
        quality_recipe,
        results,
        repository_root=tmp_path,
    )
    quality_receipt = build_receipt(
        source_identity=source_identity,
        recipe=quality_recipe,
        recipe_artifact=quality_recipe_artifact,
        run=run,
        prepared=quality_prepared,
        results=results,
        result_artifacts=quality_result_artifacts,
        batch=batch,
        execution=execution,
        batch_directory=tmp_path,
        repository_root=tmp_path,
        research_wall_seconds=1.25,
        render_wall_seconds=2.5,
    )
    assert {
        row["resultCode"]: row["qualityEvidenceHash"]
        for row in quality_receipt["results"]
    } == quality_hashes
    quality_receipt_path = tmp_path / "quality-five-video-evidence.json"
    quality_receipt_path.write_text(json.dumps(quality_receipt), encoding="utf-8")
    quality_audit = validate_existing_evidence(
        quality_receipt_path, repository_root=tmp_path
    )
    assert quality_audit["status"] == "passed"
    assert quality_audit["validationClass"] == "current_intelligent"
    assert quality_audit["publicationEligible"] is True

    # A rehashed file envelope cannot promote an incomplete or incompatible
    # result when evidence is reopened independently of receipt creation.
    quality_result_path = tmp_path / quality_receipt["results"][0]["artifact"]["path"]
    original_result = quality_result_path.read_bytes()
    for field, key, value in (
        (None, "disposition", "rejected"),
        ("runtimeCompatibility", "status", "failed"),
        ("proofOutcome", "completionStatus", "incomplete"),
        ("qualification", "allDecisionPolicyAuditComplete", False),
        ("qualification", "playingQualityEvidenceHash", "0" * 64),
    ):
        changed = json.loads(original_result)
        (changed if field is None else changed[field])[key] = value
        content = json.dumps(changed).encode("utf-8")
        quality_result_path.write_bytes(content)
        envelope = json.loads(json.dumps(quality_receipt))
        envelope["results"][0]["artifact"].update(
            sha256=hashlib.sha256(content).hexdigest(), bytes=len(content)
        )
        envelope["sha256"] = canonical_sha256(
            {key: value for key, value in envelope.items() if key != "sha256"}
        )
        altered_path = tmp_path / "nonqualifying-video-evidence.json"
        altered_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ValueError, match="result authority identity"):
            validate_existing_evidence(altered_path, repository_root=tmp_path)
    quality_result_path.write_bytes(original_result)

    for field, value in (
        ("proofCompletionStatus", "incomplete"),
        ("proofAchievedMode", "unrecognized"),
        ("proofClaimSatisfied", False),
    ):
        envelope = json.loads(json.dumps(quality_receipt))
        envelope["results"][0][field] = value
        envelope["sha256"] = canonical_sha256(
            {key: value for key, value in envelope.items() if key != "sha256"}
        )
        altered_path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ValueError, match="result authority identity"):
            validate_existing_evidence(altered_path, repository_root=tmp_path)

    substituted_receipt = json.loads(json.dumps(quality_receipt))
    substituted_receipt["recipeArtifact"] = recipe_artifact
    substituted_receipt["sha256"] = canonical_sha256(
        {key: value for key, value in substituted_receipt.items() if key != "sha256"}
    )
    substituted_path = tmp_path / "substituted-five-video-evidence.json"
    substituted_path.write_text(json.dumps(substituted_receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="recipe authority identity"):
        validate_existing_evidence(substituted_path, repository_root=tmp_path)

    invalid_prepared = json.loads(json.dumps(quality_prepared))
    invalid_prepared["items"][0]["source"]["auditOnly"] = True
    invalid_prepared["items"][0]["source"]["publicationEligible"] = False
    invalid_prepared["items"][0]["source"]["renderPurpose"] = "research_audit"
    invalid_prepared["sha256"] = canonical_sha256(
        {key: value for key, value in invalid_prepared.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="intelligent-play evidence"):
        build_receipt(
            source_identity=source_identity,
            recipe=quality_recipe,
            recipe_artifact=quality_recipe_artifact,
            run=run,
            prepared=invalid_prepared,
            results=results,
            result_artifacts=quality_result_artifacts,
            batch=batch,
            execution=execution,
            batch_directory=tmp_path,
            repository_root=tmp_path,
            research_wall_seconds=1.25,
            render_wall_seconds=2.5,
        )

    result_path = tmp_path / result_artifacts[result_codes[0]]["path"]
    result_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable Research artifact"):
        persist_research_documents(
            tmp_path / "evidence",
            recipe,
            results,
            repository_root=tmp_path,
        )

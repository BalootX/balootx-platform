from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from baloot_platform.challenge.catalog_control import (
    FoundryCatalogServiceV1,
    SQLiteFoundryCatalogStoreV1,
)
from baloot_platform.challenge.context_search import CriticalContextSearchResultV1
from baloot_platform.challenge.evidence_v3 import canonical_sha256
from baloot_platform.challenge.sakka_context import (
    ContextualizedLaffaV1,
    CuratedSakkaSignerV1,
    SakkaContextV1,
)
from baloot_platform.challenge.sakka_generation import (
    NaturalSakkaGenerationProfileV1,
    NaturalSakkaGeneratorV1,
)
from baloot_platform.challenge.score_reachability import (
    ScoreReachabilityCatalogFactoryV1,
    ScoreReachabilityProfileV1,
)
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.challenge.trace_import import (
    TraceImportCatalogFactoryV1,
    TraceImportPolicyV1,
)
from baloot_platform.core.enums import Seat
from baloot_platform.storage.artifact_store import LocalContentAddressedArtifactStore


def _service(root, tmp_path):
    artifacts = LocalContentAddressedArtifactStore(tmp_path / "artifacts")
    simulator = NaturalLaffaSimulatorV1.from_root(root)
    trace_factory = TraceImportCatalogFactoryV1(
        artifacts,
        simulator.rules,
        bundle_hash=simulator.bundle_hash,
        engine_version=simulator.engine_version,
        policy=TraceImportPolicyV1.build(
            {
                "schemaVersion": 1,
                "maximumEntries": 100,
                "allowedRightsStatuses": ["cleared", "research_only"],
                "historicalVerificationStatuses": ["verified_action_order"],
                "expertVerificationStatuses": [
                    "unreviewed",
                    "verified_expert",
                    "consensus_approved",
                ],
                "requireCompleteHiddenState": True,
            }
        ),
    )
    score_factory = ScoreReachabilityCatalogFactoryV1(
        simulator.rules,
        ScoreReachabilityProfileV1.build(
            {
                "schemaVersion": 1,
                "id": "foundry_catalog_reachability_test_v1",
                "revision": "1",
                "maximumWitnessTraces": 100,
                "witnessesPerOutcome": 4,
                "maximumRounds": 20,
                "maximumStates": 100000,
            }
        ),
    )
    store = SQLiteFoundryCatalogStoreV1(tmp_path / "foundry-catalogs.sqlite3")
    return (
        simulator,
        artifacts,
        FoundryCatalogServiceV1(
            store,
            artifacts,
            trace_factory,
            score_factory,
            CuratedSakkaSignerV1.from_secret(
                b"foundry-catalog-curated-sakka-test-secret-000001"
            ),
        ),
    )


def _played(simulator, seed, dealer):
    result = simulator.simulate(seed, dealer=dealer, match_score=(0, 0), round_number=1)
    assert result.publishable_laffa
    return result.final_trace


def _reference(artifacts, trace):
    value = artifacts.put(
        json.dumps(
            trace.to_dict(trusted=True),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        media_type="application/vnd.balootx.complete-trace-v1+json",
        name=f"{trace.sha256}.json",
    )
    return {"ownerKind": "complete_trace_v1", "ownerHash": trace.sha256, **value}


def _historical_entry(reference):
    return {
        "ordinal": 0,
        "traceReference": reference,
        "provenance": {
            "schemaVersion": 1,
            "lane": "historical",
            "sourceRecordId": "historical-source-001",
            "sourceEvidenceHash": "d" * 64,
            "rightsStatus": "cleared",
            "verificationStatus": "verified_action_order",
            "completeHiddenStateAttested": True,
        },
    }


def _complete_match(simulator, *, seed=1000):
    traces = []
    score = (0, 0)
    dealer = Seat.EAST
    round_number = 1
    while max(score) < 152 and len(traces) < 20:
        result = simulator.simulate(
            seed,
            dealer=dealer,
            match_score=score,
            round_number=round_number,
        )
        assert result.publishable_laffa
        trace = result.final_trace
        traces.append(trace)
        score = tuple(trace.trusted_final_state["matchScore"])
        dealer = Seat.parse(trace.trusted_final_state["dealer"]).next()
        round_number = int(trace.trusted_final_state["roundNumber"]) + 1
        seed += 1
    assert max(score) >= 152
    return tuple(traces)


def test_trace_import_catalog_is_immutable_persisted_and_reloadable(root, tmp_path):
    simulator, artifacts, service = _service(root, tmp_path)
    trace = _played(simulator, 1, Seat.EAST)
    entry = _historical_entry(_reference(artifacts, trace))
    first = service.create_trace_import_catalog("historical", (entry,))
    second = service.create_trace_import_catalog("historical", (entry,))
    assert first == second
    identity = first["record"]["catalogHash"]
    loaded = service.load_trace_import_catalog(identity)
    assert loaded.sha256 == identity
    assert loaded.entries[0].trace_reference["ownerHash"] == trace.sha256
    listing = service.list(kind="trace_import_catalog_v1", limit=10)
    assert [item["catalogHash"] for item in listing["items"]] == [identity]
    assert listing["nextCursor"] is None


def test_reachability_catalog_and_proof_rebuild_from_content_addressed_witnesses(
    root, tmp_path
):
    simulator, artifacts, service = _service(root, tmp_path)
    east = _played(simulator, 1, Seat.EAST)
    north = _played(simulator, 2, Seat.NORTH)
    references = (_reference(artifacts, east), _reference(artifacts, north))
    created = service.create_score_reachability_catalog(references)
    catalog_hash = created["record"]["catalogHash"]
    goal = tuple(
        int(east.trusted_final_state["roundScore"]["awardedPoints"][index])
        + int(north.trusted_final_state["roundScore"]["awardedPoints"][index])
        for index in range(2)
    )
    first = service.prove_score_context(
        catalog_hash,
        goal,
        initial_dealer="east",
        target_dealer="west",
    )
    second = service.prove_score_context(
        catalog_hash,
        goal,
        initial_dealer="east",
        target_dealer="west",
    )
    assert first == second
    proof_hash = first["record"]["catalogHash"]
    reloaded = service.get(proof_hash)
    assert reloaded == first
    assert reloaded["document"]["proof"]["targetScore"] == list(goal)
    assert reloaded["document"]["verifiedContext"]["context"]["dealer"] == "west"
    assert (
        len({item["dealCommitment"] for item in reloaded["document"]["proof"]["steps"]})
        == 2
    )
    verified = service.load_verified_score_context(proof_hash)
    assert verified.sha256 == reloaded["document"]["verifiedContext"]["sha256"]

    fresh_store = SQLiteFoundryCatalogStoreV1(tmp_path / "foundry-catalogs.sqlite3")
    fresh_service = FoundryCatalogServiceV1(
        fresh_store,
        artifacts,
        service.trace_import_factory,
        service.score_reachability_factory,
    )
    assert (
        fresh_service.prove_score_context(
            catalog_hash,
            goal,
            initial_dealer="east",
            target_dealer="west",
        )
        == first
    )


def test_catalog_service_rejects_trace_owner_tampering(root, tmp_path):
    simulator, artifacts, service = _service(root, tmp_path)
    trace = _played(simulator, 3, Seat.EAST)
    reference = {**_reference(artifacts, trace), "ownerHash": "0" * 64}
    with pytest.raises(ValueError, match="owner identity"):
        service.create_score_reachability_catalog((reference,))


def test_natural_and_curated_sakkas_are_immutable_reloadable_artifacts(root, tmp_path):
    simulator, artifacts, service = _service(root, tmp_path)
    descendants = _complete_match(simulator)
    references = tuple(_reference(artifacts, trace) for trace in descendants)

    natural = service.create_natural_sakka(references, target=152)
    natural_hash = natural["record"]["catalogHash"]
    assert service.load_sakka(natural_hash).sha256 == natural_hash

    selections = []
    for index, descendant in enumerate(descendants):
        start = descendant.trusted_final_state["generation"]
        parent = simulator.simulate(
            int(start["rootSeed"]),
            dealer=Seat.parse(start["dealer"]),
            match_score=(0, 0),
            round_number=1,
        ).final_trace
        assert parent.trusted_final_state["phase"] == "scored"
        context = SakkaContextV1.build(
            starting_score=tuple(start["matchScore"]),
            dealer=start["dealer"],
            round_number=int(start["roundNumber"]),
            target=152,
            reachability_mode="verified_enumeration",
            reachability_proof_hash=f"{index + 1:064x}",
        )
        parent_decision = next(
            item
            for item in parent.decision_ledger.records
            if item.decision_kind == "play" and not item.forced
        )
        descendant_decision = next(
            item
            for item in descendant.decision_ledger.records
            if item.decision_kind == "play" and not item.forced
        )
        contextualized = ContextualizedLaffaV1.build(
            parent=parent,
            descendant=descendant,
            context=context,
            original_decision_id=parent_decision.decision_id,
            descendant_decision_id=descendant_decision.decision_id,
            analysis_hash=f"{index + 100:064x}",
            context_amplification={"matchProbabilitySwing": 10.0},
            strategy_changed=(
                parent_decision.chosen_action != descendant_decision.chosen_action
            ),
            recommendation_stable=True,
        )
        evaluated = (
            {
                "contextHash": context.sha256,
                "accepted": True,
                "contextualizedLaffaHash": contextualized.sha256,
            },
        )
        core = {
            "schemaVersion": 1,
            "parentTraceHash": parent.sha256,
            "originalDecisionId": parent_decision.decision_id,
            "profileHash": "a" * 64,
            "analysisProfileHash": "b" * 64,
            "evaluated": list(evaluated),
            "acceptedHashes": [contextualized.sha256],
            "selectedHash": contextualized.sha256,
        }
        result = CriticalContextSearchResultV1(
            parent.sha256,
            parent_decision.decision_id,
            "a" * 64,
            "b" * 64,
            evaluated,
            (contextualized,),
            contextualized.sha256,
            canonical_sha256(core),
            (descendant,),
        )
        persisted = service.persist_context_search_result(result)
        selections.append(
            {
                "contextSearchResultHash": persisted["record"]["catalogHash"],
                "contextualizedLaffaHash": contextualized.sha256,
            }
        )

    curated = service.create_curated_sakka(selections, target=152)
    curated_hash = curated["record"]["catalogHash"]
    reloaded = service.load_sakka(curated_hash)
    assert reloaded.composition_hash == curated_hash
    assert reloaded.verify(service.sakka_signer)
    assert (
        service.list_sakkas(provenance="natural")["items"][0]["catalogHash"]
        == natural_hash
    )
    assert (
        service.list_sakkas(provenance="curated")["items"][0]["catalogHash"]
        == curated_hash
    )


def test_generated_natural_sakka_retains_every_auction_attempt(root, tmp_path):
    simulator, _artifacts, service = _service(root, tmp_path)
    profile = NaturalSakkaGenerationProfileV1.build(
        {
            "schemaVersion": 1,
            "id": "catalog-natural-sakka-generation-test-v1",
            "revision": "1",
            "target": 152,
            "maximumRounds": 32,
            "seedPolicy": "sha256_round_ordinal_v1",
        }
    )
    result = NaturalSakkaGeneratorV1(simulator, profile).generate(5150)
    persisted = service.persist_natural_sakka_generation(result)
    generation_hash = persisted["record"]["catalogHash"]
    loaded = service.load_natural_sakka_generation(generation_hash)

    assert loaded.sha256 == result.sha256
    assert loaded.sakka.sha256 == result.sakka.sha256
    document = service.get(generation_hash)["document"]
    groups = document["attemptTraceReferenceGroups"]
    assert [len(group) for group in groups] == [
        len(item.attempt_traces) for item in result.rounds
    ]
    assert all(
        group[-1]["ownerHash"] == item.final_trace.sha256
        for group, item in zip(groups, result.rounds)
    )


def test_trusted_admin_catalog_api_persists_and_proves_exact_context(
    root, tmp_path, monkeypatch
):
    from baloot_platform.service import api

    simulator, artifacts, service = _service(root, tmp_path)
    monkeypatch.setattr(api, "FOUNDRY_CATALOGS", service)
    east = _played(simulator, 1, Seat.EAST)
    north = _played(simulator, 2, Seat.NORTH)
    east_reference = _reference(artifacts, east)
    north_reference = _reference(artifacts, north)
    goal = tuple(
        int(east.trusted_final_state["roundScore"]["awardedPoints"][index])
        + int(north.trusted_final_state["roundScore"]["awardedPoints"][index])
        for index in range(2)
    )
    with TestClient(api.app) as client:
        imported = client.post(
            "/api/v2/trace-import-catalogs",
            json={"lane": "historical", "entries": [_historical_entry(east_reference)]},
        )
        assert imported.status_code == 201, imported.text
        reachability = client.post(
            "/api/v2/score-reachability-catalogs",
            json={"traceReferences": [east_reference, north_reference]},
        )
        assert reachability.status_code == 201, reachability.text
        catalog_hash = reachability.json()["record"]["catalogHash"]
        proof = client.post(
            f"/api/v2/score-reachability-catalogs/{catalog_hash}:prove",
            json={
                "targetScore": list(goal),
                "targetDealer": "west",
                "initialDealer": "east",
            },
        )
        assert proof.status_code == 201, proof.text
        proof_hash = proof.json()["record"]["catalogHash"]
        assert client.get(f"/api/v2/foundry-catalogs/{proof_hash}").status_code == 200
        listing = client.get(
            "/api/v2/foundry-catalogs",
            params={"kind": "score_reachability_proof_v1"},
        )
        assert listing.status_code == 200
        assert [item["catalogHash"] for item in listing.json()["items"]] == [proof_hash]


def test_postgres_catalog_migration_declares_immutable_index(root):
    sql = (
        root
        / "src"
        / "baloot_platform"
        / "resources"
        / "migrations"
        / "018_foundry_catalog_control.sql"
    ).read_text(encoding="utf-8")
    assert "foundry_catalogs_v1" in sql
    assert "catalog_hash CHAR(64) PRIMARY KEY" in sql
    assert "metadata_json JSONB NOT NULL" in sql


def test_qualified_laffa_library_is_immutable_and_durable(root, tmp_path):
    _simulator, artifacts, service = _service(root, tmp_path)
    bundle_hash = "a" * 64
    bundle = artifacts.put(
        b'{"schemaVersion":1}',
        media_type="application/vnd.balootx.render-bundle-v1+json",
        name="qualified.render-bundle.json",
    )
    prepared = {
        "traceHash": "b" * 64,
        "candidateHash": "c" * 64,
        "shortCode": "BALXTEST00000001",
        "basename": "balootx-test-laffa",
        "bundleArtifact": {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle_hash,
            **bundle,
        },
        "bundleHash": bundle_hash,
        "decisionKind": "play",
        "selectionScore": 88.0,
        "laffaSelectionScore": {
            "schemaVersion": 1,
            "scoreFamily": "laffa_content_selection",
            "profileHash": "e" * 64,
            "evidenceHash": "f" * 64,
            "score": 91.0,
            "passed": True,
            "rejectionReasons": [],
            "sha256": "1" * 64,
        },
        "confidence": 1.0,
        "truthLabel": "exact_optimal",
        "laffaClassification": "single_perspective_masterpiece",
        "interestingProfiles": ["only_winning_move", "contract_rescue"],
        "sourceCampaignHash": "2" * 64,
        "qualification": {
            "schemaVersion": 1,
            "localPreviewReady": True,
            "publicationReady": False,
            "publicationBlockers": ["professional_agent_qualification_missing"],
        },
    }
    missing_laffa_score = dict(prepared)
    missing_laffa_score.pop("laffaSelectionScore")
    with pytest.raises(ValueError, match="content-selection score is missing"):
        service.put_qualified_laffa(missing_laffa_score)
    created = service.put_qualified_laffa(prepared)
    service.put_qualified_laffa(prepared)

    library = service.list_qualified_laffat(limit=10)
    assert created["record"]["kind"] == "qualified_laffa_v2"
    assert created["document"]["schemaVersion"] == 2
    assert library["items"] == [prepared]
    assert library["nextCursor"] is None


def test_qualified_laffa_library_supports_filtered_sorted_cursor_queries(root, tmp_path):
    _simulator, artifacts, service = _service(root, tmp_path)

    def prepared_item(index: int, *, score: float, truth: str, classification: str, profile: str):
        bundle_hash = f"{index:x}" * 64
        bundle = artifacts.put(
            b'{"schemaVersion":1}',
            media_type="application/vnd.balootx.render-bundle-v1+json",
            name=f"query-{index}.render-bundle.json",
        )
        return {
            "traceHash": f"{index + 2:x}" * 64,
            "candidateHash": f"{index + 4:x}" * 64,
            "shortCode": f"BALXQUERY{index:08d}",
            "basename": f"balootx-query-{index}",
            "bundleArtifact": {"ownerKind": "render_bundle_v1", "ownerHash": bundle_hash, **bundle},
            "bundleHash": bundle_hash,
            "decisionKind": "play",
            "selectionScore": score - 5,
            "laffaSelectionScore": {
                "schemaVersion": 1,
                "scoreFamily": "laffa_content_selection",
                "profileHash": "a" * 64,
                "evidenceHash": f"{index + 6:x}" * 64,
                "score": score,
                "passed": True,
                "rejectionReasons": [],
                "sha256": f"{index + 8:x}" * 64,
            },
            "confidence": score / 100,
            "truthLabel": truth,
            "laffaClassification": classification,
            "interestingProfiles": [profile],
            "sourceCampaignHash": "f" * 64,
            "qualification": {"schemaVersion": 1, "localPreviewReady": True},
        }

    first = prepared_item(1, score=92, truth="exact_optimal", classification="single_perspective_masterpiece", profile="only_winning_move")
    second = prepared_item(2, score=78, truth="robust_best_known", classification="two_seat_duel", profile="small_card_big_gain")
    service.put_qualified_laffa(second)
    legacy_identity = canonical_sha256(
        {
            "bundleHash": first["bundleHash"],
            "laffaSelectionScoreHash": first["laffaSelectionScore"]["sha256"],
        }
    )
    service._persist(
        identity=legacy_identity,
        kind="qualified_laffa_v2",
        document={"schemaVersion": 2, "preparedItem": first},
        media_type="application/vnd.balootx.qualified-laffa-v2+json",
        metadata={"shortCode": first["shortCode"]},
    )
    # Re-indexing an immutable historical item repairs only its rebuildable
    # query projection; the catalog record and evidence document stay intact.
    service.put_qualified_laffa(first)

    page_one = service.list_qualified_laffat(sort="laffa_score", direction="desc", limit=1)
    assert page_one["items"] == [first]
    assert page_one["nextCursor"]
    page_two = service.list_qualified_laffat(
        sort="laffa_score", direction="desc", cursor=page_one["nextCursor"], limit=1
    )
    assert page_two["items"] == [second]
    assert service.list_qualified_laffat(truth_label="robust_best_known")["items"] == [second]
    assert service.list_qualified_laffat(classification="two_seat_duel")["items"] == [second]
    assert service.list_qualified_laffat(profile="small_card_big_gain")["items"] == [second]
    assert service.list_qualified_laffat(search="QUERY00000001")["items"] == [first]
    with pytest.raises(ValueError, match="does not match"):
        service.list_qualified_laffat(
            sort="laffa_score", direction="asc", cursor=page_one["nextCursor"], limit=1
        )


def test_qualified_laffa_library_rejects_and_hides_incomplete_legacy_items(
    root, tmp_path
):
    _simulator, artifacts, service = _service(root, tmp_path)
    bundle_hash = "d" * 64
    bundle = artifacts.put(
        b'{"schemaVersion":1}',
        media_type="application/vnd.balootx.render-bundle-v1+json",
        name="legacy.render-bundle.json",
    )
    legacy = {
        "traceHash": "e" * 64,
        "candidateHash": "f" * 64,
        "shortCode": "BALXLEGACY000001",
        "basename": "balootx-legacy-laffa",
        "bundleArtifact": {
            "ownerKind": "render_bundle_v1",
            "ownerHash": bundle_hash,
            **bundle,
        },
        "bundleHash": bundle_hash,
        "qualification": {"localPreviewReady": True},
    }
    with pytest.raises(ValueError, match="decisionKind"):
        service.put_qualified_laffa(legacy)

    service._persist(
        identity=bundle_hash,
        kind="qualified_laffa_v1",
        document={"schemaVersion": 1, "preparedItem": legacy},
        media_type="application/vnd.balootx.qualified-laffa-v1+json",
        metadata={"shortCode": legacy["shortCode"]},
    )
    assert service.list_qualified_laffat(limit=10)["items"] == []


def test_all_declared_sdks_expose_foundry_catalog_resources(root):
    python_client = (root / "sdk" / "python" / "client.py").read_text(encoding="utf-8")
    typescript_client = (root / "sdk" / "typescript" / "src" / "client.ts").read_text(
        encoding="utf-8"
    )
    go_client = (root / "sdk" / "go" / "client.go").read_text(encoding="utf-8")
    for value in (python_client, typescript_client, go_client):
        assert "/api/v2/trace-import-catalogs" in value
        assert "/api/v2/score-reachability-catalogs" in value
        assert "/api/v2/foundry-catalogs" in value
        assert "/api/v2/analysis-batches" in value
        assert "/api/v2/campaign-equivalence-proofs" in value
        assert "/api/v2/context-searches" in value
        assert "/api/v2/sakkas" in value
        assert "/api/v2/sakkas/natural" in value
        assert "/api/v2/sakkas/curated" in value
        assert "/api/v2/sakka-generations" in value
        assert "/api/v2/foundry-profile" in value
        assert "/api/v2/campaigns" in value
        assert "/attempts" in value


def test_all_declared_sdks_expose_duplicate_challenge_resources(root):
    clients = (
        (root / "sdk" / "python" / "client.py").read_text(encoding="utf-8"),
        (root / "sdk" / "typescript" / "src" / "client.ts").read_text(encoding="utf-8"),
        (root / "sdk" / "go" / "client.go").read_text(encoding="utf-8"),
    )
    for value in clients:
        assert "/api/v2/challenge-locators" in value
        assert "/attempts" in value
        assert ":revoke" in value

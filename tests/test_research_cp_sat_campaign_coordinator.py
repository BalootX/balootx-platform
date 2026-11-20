from __future__ import annotations

import hashlib
import threading
from types import SimpleNamespace

import pytest

from ortools.sat.python import cp_model

from baloot_platform.research import campaign_coordinator as coordinator
from baloot_platform.research.campaign_coordinator import (
    CampaignBudgetV1,
    CampaignCancellationTokenV1,
    CampaignShardOutcomeV1,
    CampaignShardRequestV1,
    CampaignStageTimingV1,
    CampaignVariantV1,
    DeterministicCpSatCampaignCoordinatorV1,
    RetryableCampaignShardError,
    build_rules_preserving_diversity_document_v1,
    decision_state_card_vector_v1,
    decision_state_card_hamming_distance_v1,
    rules_preserving_symmetry_key_v1,
    semantic_threshold_distance_v1,
    select_campaign_variants_v1,
)
from baloot_platform.research.contracts import canonical_json, canonical_sha256
from baloot_platform.research.ortools_research_executor import (
    _apply_candidate_partition_v1,
)
from baloot_platform.research.sharding import partition_research_shards


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_candidate_partitions_are_disjoint_and_cover_one_fixed_vector():
    statuses = []
    for residue in range(5):
        model = cp_model.CpModel()
        variables = tuple(model.new_int_var(0, 31, f"card_{index}") for index in range(4))
        for variable, value in zip(variables, (1, 2, 3, 4), strict=True):
            model.add(variable == value)
        _apply_candidate_partition_v1(model, variables, (5, residue, 1), name="test")
        statuses.append(cp_model.CpSolver().solve(model))

    assert sum(status in {cp_model.FEASIBLE, cp_model.OPTIMAL} for status in statuses) == 1


def _shards(count: int, *, attempts: int = 3):
    return partition_research_shards(
        run_code="RUN-CAMPAIGN0001",
        run_hash=_hash("run"),
        recipe_hash=_hash("recipe"),
        seed_start=100,
        seed_count=count,
        seeds_per_shard=1,
        maximum_attempts=attempts,
    )


def _budget(*, results: int, workers: int = 3, wall: float = 2.0):
    return CampaignBudgetV1(
        result_count=results,
        shard_worker_count=workers,
        maximum_wall_time_seconds=wall,
        cancellation_poll_milliseconds=5,
    )


@pytest.fixture
def campaign_clock(monkeypatch):
    now_ns = [0]
    monkeypatch.setattr(coordinator, "time", SimpleNamespace(
        **{**vars(coordinator.time),
            "perf_counter_ns": lambda: now_ns[0],
            "perf_counter": lambda: now_ns[0] / 1_000_000_000}))
    return now_ns


def _deliver_shards_in_order(monkeypatch, order):
    """Release real shard futures in a specified order at the coordinator's wait boundary."""
    executor = coordinator.ThreadPoolExecutor
    wait = coordinator.wait
    gates = {index: threading.Event() for index in order}
    started = threading.Barrier(len(order) + 1)
    pending_order = iter(order)
    submissions = {}
    first_wait = True

    class OrderedExecutor(executor):
        def submit(self, fn, solver, request):
            index = request.shard.shard_index

            def gated_solver(request):
                started.wait()
                gates[index].wait()
                return solver(request)

            future = super().submit(fn, gated_solver, request)
            submissions[index] = (future, request.cancellation)
            return future

        def shutdown(self, *args, **kwargs):
            for gate in gates.values():
                gate.set()
            return super().shutdown(*args, **kwargs)

    def ordered_wait(futures, **kwargs):
        nonlocal first_wait
        if first_wait:
            first_wait = False
            started.wait()
        if any(token.is_cancelled for _, token in submissions.values()):
            for gate in gates.values():
                gate.set()
            return wait(futures, return_when=kwargs["return_when"])
        index = next(pending_order)
        future, _ = submissions[index]
        assert future in futures
        gates[index].set()
        wait((future,))
        return {future}, set(futures) - {future}

    monkeypatch.setattr(coordinator, "ThreadPoolExecutor", OrderedExecutor)
    monkeypatch.setattr(coordinator, "wait", ordered_wait)


@pytest.mark.parametrize("checkpoint_bytes", [0, 42_000_000])
def test_private_checkpoints_survive_cancellation_without_live_disclosure(
    checkpoint_bytes, campaign_clock, monkeypatch,
):
    cancellation, release, finished = (threading.Event() for _ in range(3))
    requests = []
    wait = coordinator.wait

    def wait_for_checkpoint(futures, **kwargs):
        if requests and requests[0].cancellation.is_cancelled:
            requests[0].cancellation_acknowledgement.wait()
            campaign_clock[0] = 40_000_000
            return set(), set(futures)
        return wait(futures, **kwargs)

    monkeypatch.setattr(coordinator, "wait", wait_for_checkpoint)
    live = []
    document = {"kind": "quality_bid_incumbent_checkpoint_v1", "stage": "bid", "artifact": {"privateHand": [1, 2]},
                "sourcePrefixHash": _hash("prefix")}
    expected = {"kind": "quality_bid_incumbent_checkpoint_v1", "stage": "bid", "artifact": {"privateHand": [1, 2]},
                "sourcePrefixHash": _hash("prefix")}
    if checkpoint_bytes:
        document["artifact"]["payload"] = "x" * checkpoint_bytes
        expected["artifact"]["payload"] = document["artifact"]["payload"]
    serialized = canonical_json(document)
    def solver(request):
        requests.append(request)
        try:
            request.report_worker_cpu_sample(123_000_000)
            request.report_worker_cpu_sample(100_000_000)  # Older samples cannot erase work.
            request.report_progress({"currentStage": "playingQuality",
                "completedCandidates": 1, "candidateHash": _hash("candidate")})
            if checkpoint_bytes:
                cancellation.set()  # Request cancellation before ingesting the large frame.
            request.report_checkpoint(serialized if checkpoint_bytes else document)
            if checkpoint_bytes:
                request.cancellation_requested()  # The process pump polls after each frame.
            request.report_checkpoint(serialized)  # Process retransmission is idempotent.
            document["artifact"]["privateHand"].append(3)
            cancellation.set()
            request.cancellation.wait()
            request.cancellation_requested()
            release.wait()
            return CampaignShardOutcomeV1(())
        finally:
            finished.set()
    try:
        receipt = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=_shards(1), solver=solver, budget=_budget(results=1, workers=1),
            cancellation_event=cancellation, progress_callback=live.append)
        assert receipt.status == "cancelled"
        evidence = receipt.attempts[0].partial_evidence
        assert evidence["qualityCheckpoints"] == {canonical_sha256(expected): expected}
        assert evidence["workerCpuAccounting"] == {
            "cpuTimeNs": 123_000_000, "coverage": "sampled_lower_bound"}
        assert evidence["cancellation"]["acknowledged"] is True
        assert evidence["cancellation"]["latencyNs"] <= 250_000_000
        assert evidence["cancellation"]["workerTerminated"] is False
        assert all("qualityCheckpoints" not in item and "artifact" not in item for item in live)
        assert receipt.attempts[0].variant_hashes == ()
        terminal = [item for item in live if item.get("currentStage") == "candidate_terminal"]
        assert len(terminal) == 1
        assert terminal[0]["candidateStatus"] == "incomplete"
        assert terminal[0]["candidateHash"] == _hash("candidate")
    finally:
        release.set()
        finished.wait()


def test_serialized_checkpoint_does_not_repeat_coordinator_serialization(monkeypatch):
    from baloot_platform.research import campaign_coordinator as module
    progress = module.CampaignAttemptProgressV1()
    document = {"artifact": {"privateHand": [1, 2]}}
    serialized = canonical_json(document)
    def forbidden(_value):
        raise AssertionError("worker checkpoint must not be serialized twice")
    monkeypatch.setattr(module, "canonical_json", forbidden)
    progress.checkpoint(serialized)
    assert progress.snapshot()["qualityCheckpoints"] == {canonical_sha256(document): document}


def test_snapshot_waits_for_in_flight_private_checkpoint(monkeypatch):
    from baloot_platform.research import campaign_coordinator as module

    checkpoint_started, release_checkpoint, snapshot_started = (
        threading.Event() for _ in range(3))
    document = {"artifact": {"privateHand": [1, 2]}}
    original = module.CampaignAttemptProgressV1.checkpoint

    def ingest(progress, evidence):
        checkpoint_started.set()
        release_checkpoint.wait()
        original(progress, evidence)

    monkeypatch.setattr(module.CampaignAttemptProgressV1, "checkpoint", ingest)
    request = CampaignShardRequestV1(_shards(1)[0], 1, CampaignCancellationTokenV1())
    snapshots = []

    def snapshot():
        snapshot_started.set()
        snapshots.append(request.progress_snapshot())

    writer = threading.Thread(target=request.report_checkpoint, args=(document,))
    reader = threading.Thread(target=snapshot)
    writer.start()
    try:
        checkpoint_started.wait()
        acquired = request.evidence_transfer_lock.acquire(blocking=False)
        if acquired:
            request.evidence_transfer_lock.release()
        assert not acquired
        reader.start()
        snapshot_started.wait()
        assert snapshots == []
    finally:
        release_checkpoint.set()
        writer.join()
        if reader.ident is not None:
            reader.join()
    with request.evidence_transfer_lock:
        request.report_checkpoint(document)  # The process pump already owns this RLock.
    assert snapshots == [{"currentStage": "queued", "qualityCheckpoints": {
        canonical_sha256(document): document}}]


def test_out_of_order_shards_merge_by_semantic_hash_with_stable_identities(monkeypatch):
    shards = _shards(3)
    completion_order: list[int] = []

    def solver(request):
        completion_order.append(request.shard.shard_index)
        variant = CampaignVariantV1.build(
            {"seed": request.shard.seed_start, "value": request.shard.shard_index}
        )
        return CampaignShardOutcomeV1((variant,))

    with monkeypatch.context() as patch:
        _deliver_shards_in_order(patch, (2, 1, 0))
        parallel = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=shards,
            solver=solver,
            budget=_budget(results=2, workers=3),
        )
    serial = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=lambda request: CampaignShardOutcomeV1(
            (
                CampaignVariantV1.build(
                    {
                        "seed": request.shard.seed_start,
                        "value": request.shard.shard_index,
                    }
                ),
            )
        ),
        budget=_budget(results=2, workers=1),
    )

    assert completion_order[:3] == [2, 1, 0]
    assert parallel.status == serial.status == "completed"
    assert [item.semantic_hash for item in parallel.accepted_variants] == [
        item.semantic_hash for item in serial.accepted_variants
    ]
    assert parallel.semantic_sha256 == serial.semantic_sha256
    assert parallel.to_dict()["workerDeclaration"]["cpSatWorkersPerSolve"] == 1
    assert {
        item.cp_sat_workers_per_solve for item in (parallel.budget, serial.budget)
    } == {1}


def test_semantic_receipt_excludes_run_specific_shard_identity():
    first = _shards(2)
    second = partition_research_shards(
        run_code="RUN-CAMPAIGN0002",
        run_hash=_hash("another-run"),
        recipe_hash=first[0].recipe_hash,
        seed_start=100,
        seed_count=2,
        seeds_per_shard=1,
        maximum_attempts=3,
    )

    def solve(request):
        return CampaignShardOutcomeV1(
            (CampaignVariantV1.build({"seed": request.shard.seed_start}),)
        )

    first_receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=first, solver=solve, budget=_budget(results=2, workers=2)
    )
    second_receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=second, solver=solve, budget=_budget(results=2, workers=2)
    )

    assert first_receipt.semantic_sha256 == second_receipt.semantic_sha256
    assert first_receipt.receipt_sha256 != second_receipt.receipt_sha256


def test_retry_receipt_is_exact_and_does_not_change_accepted_identity():
    shards = _shards(2, attempts=2)
    calls: dict[int, int] = {}
    lock = threading.Lock()

    def candidate(request):
        return CampaignVariantV1.build({"seed": request.shard.seed_start})

    def flaky(request):
        with lock:
            calls[request.shard.shard_index] = (
                calls.get(request.shard.shard_index, 0) + 1
            )
            call = calls[request.shard.shard_index]
        if request.shard.shard_index == 0 and call == 1:
            request.report_progress({"currentStage": "playingQuality", "completedCandidates": 1})
            raise RetryableCampaignShardError("temporary_fixture_failure")
        return CampaignShardOutcomeV1((candidate(request),))

    retried = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards, solver=flaky, budget=_budget(results=2, workers=2)
    )
    clean = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=lambda request: CampaignShardOutcomeV1((candidate(request),)),
        budget=_budget(results=2, workers=2),
    )

    assert [
        (item.shard_index, item.attempt, item.status) for item in retried.attempts
    ] == [
        (0, 1, "retryable_failure"),
        (0, 2, "completed"),
        (1, 1, "completed"),
    ]
    assert retried.attempts[0].error_code == "temporary_fixture_failure"
    assert retried.attempts[0].partial_evidence["completedCandidates"] == 1
    assert retried.semantic_sha256 == clean.semantic_sha256
    assert retried.receipt_sha256 == canonical_sha256(
        {
            key: value
            for key, value in retried.to_dict().items()
            if key != "receiptSha256"
        }
    )


@pytest.mark.parametrize("cancelled", (False, True))
def test_worker_exception_progress_distinguishes_requested_cancellation(cancelled):
    cancellation = CampaignCancellationTokenV1()
    if cancelled:
        cancellation._cancel()
    request = CampaignShardRequestV1(_shards(1)[0], 1, cancellation)

    def stopped(_request):
        raise RuntimeError("campaign shard process cancelled")

    execution = DeterministicCpSatCampaignCoordinatorV1._invoke(stopped, request)

    assert execution.outcome is None
    assert execution.error_code == "builtins.RuntimeError"
    assert request.progress_snapshot()["currentStage"] == (
        "cancelled" if cancelled else "worker_failed"
    )


def test_worker_failure_preserves_actionable_receipt_evidence():
    live = []
    checkpoint = {"artifact": {"privateHand": [1, 2]}, "sourcePrefixHash": _hash("prefix")}

    def broken(request):
        request.report_checkpoint(checkpoint)
        request.report_progress({"currentStage": "playingQuality", "completedCandidates": 1})
        raise ValueError("remaining ownership is not bound")

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=_shards(1),
        solver=broken,
        budget=_budget(results=1, workers=1),
        progress_callback=live.append,
    )

    attempt = receipt.attempts[0].to_dict()
    assert receipt.status == "incomplete"
    assert attempt["status"] == "failed"
    assert attempt["errorCode"] == "builtins.ValueError"
    assert attempt["partialEvidence"]["currentStage"] == "worker_failed"
    assert attempt["partialEvidence"]["errorCode"] == "builtins.ValueError"
    assert attempt["partialEvidence"]["errorDetail"] == (
        "remaining ownership is not bound"
    )
    assert attempt["partialEvidence"]["completedCandidates"] == 1
    assert attempt["partialEvidence"]["qualityCheckpoints"] == {canonical_sha256(checkpoint): checkpoint}
    assert all("qualityCheckpoints" not in item and "artifact" not in item for item in live)


def test_stage_updates_preserve_attempt_counts_without_stale_solver_evidence():
    from baloot_platform.research.ortools_research_executor import OrToolsResearchCampaignResultV1
    from baloot_platform.research.service import _project_ortools_campaign_statistics_run

    live = []

    def solver(request):
        assert "completedCandidates" not in request.progress_snapshot()
        request.report_progress({
            "currentStage": "playingQuality", "completedCandidates": 2,
            "candidateHash": _hash("candidate"), "qualitySolver": {"queryCount": 7},
        })
        request.report_progress({"currentStage": "construction"})
        snapshot = request.progress_snapshot()
        assert snapshot["completedCandidates"] == 2
        assert "candidateHash" not in snapshot and "qualitySolver" not in snapshot
        request.report_progress({"currentStage": "shard_budget_exhausted", "completedCandidates": 0})
        return CampaignShardOutcomeV1((), partial_evidence=request.progress_snapshot())

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=_shards(2, attempts=1), solver=solver,
        budget=_budget(results=1, workers=1), progress_callback=live.append,
    )
    assert len(receipt.attempts) == 2
    for attempt in receipt.attempts:
        assert attempt.status == "completed"
        evidence = attempt.partial_evidence
        assert evidence["completedCandidates"] == 2
        assert evidence["shardId"] == attempt.shard_id
        assert evidence["attempt"] == attempt.attempt
    assert receipt.receipt_sha256 == canonical_sha256({
        key: value for key, value in receipt.to_dict().items() if key != "receiptSha256"
    })
    assert all("qualityCheckpoints" not in item for item in live)
    campaign = OrToolsResearchCampaignResultV1(
        "incomplete", (), (), (), (), 1, 4, "rules_preserving_v1",
        "decision_state_card_hamming_v1", 1, 0.1, 0.1, receipt,
    )
    run = {
        "scenarioExecutorIdentity": {"backendIdentity": {"backendId": "ortools_cp_sat_v3"}},
        "attemptedCandidates": 4, "resultCount": 0,
    }
    projected = _project_ortools_campaign_statistics_run(run, campaign.to_dict())
    assert projected["attemptedCandidates"] == 4
    assert campaign.to_dict()["completedCandidates"] == 0


def test_duplicate_and_symmetry_hooks_reject_in_canonical_order():
    shards = _shards(3)
    duplicate = CampaignVariantV1.build({"name": "same"})
    variants = {
        0: (
            duplicate,
            CampaignVariantV1.build({"name": "a"}, symmetry_document={"group": "g"}),
        ),
        1: (duplicate,),
        2: (CampaignVariantV1.build({"name": "b"}, symmetry_document={"group": "g"}),),
    }
    coordinator = DeterministicCpSatCampaignCoordinatorV1(
        symmetry_key=lambda variant: variant.symmetry_document.get(
            "group", variant.symmetry_hash
        ),
        symmetry_policy_id="fixture_group_v1",
    )
    receipt = coordinator.run(
        shards=shards,
        solver=lambda request: CampaignShardOutcomeV1(
            variants[request.shard.shard_index]
        ),
        budget=_budget(results=5),
    )

    assert receipt.status == "not_found_within_budget"
    assert len(receipt.accepted_variants) == 2
    assert [item.rejection_reason for item in receipt.variants].count("duplicate") == 1
    assert [item.rejection_reason for item in receipt.variants].count("symmetry") == 1
    assert [item.semantic_hash for item in receipt.accepted_variants] == sorted(
        item.semantic_hash for item in receipt.accepted_variants
    )


def test_diversity_rejection_runs_one_deterministic_fallback_in_parallel():
    shards = partition_research_shards(
        run_code="RUN-REPLENISH001",
        run_hash=_hash("replenish-run"),
        recipe_hash=_hash("replenish-recipe"),
        seed_start=10,
        seed_count=4,
        seeds_per_shard=2,
    )
    calls: list[tuple[int, int, int]] = []

    def solver(request):
        calls.append(
            (
                request.shard.shard_index,
                request.attempt,
                len(request.accepted_variants),
            )
        )
        replacement = request.attempt > 1
        return CampaignShardOutcomeV1(
            (
                CampaignVariantV1.build(
                    {
                        "shard": request.shard.shard_index,
                        "candidate": request.attempt,
                    },
                    symmetry_document={
                        "group": (
                            f"replacement-{request.shard.shard_index}"
                            if replacement
                            else "same"
                        )
                    },
                ),
            )
        )

    receipt = DeterministicCpSatCampaignCoordinatorV1(
        symmetry_key=lambda variant: variant.symmetry_document["group"],
        symmetry_policy_id="fixture_group_v1",
    ).run(
        shards=shards,
        solver=solver,
        budget=_budget(results=2, workers=2),
        replenish_diversity_rejections=True,
    )

    assert receipt.status == "completed"
    assert len(receipt.accepted_variants) == 2
    assert len(calls) == 4
    assert sorted(attempt for _shard, attempt, _accepted in calls) == [1, 1, 2, 2]
    assert sorted(accepted for _shard, _attempt, accepted in calls) == [0, 0, 1, 1]
    assert {
        shard % 2 for shard, attempt, _accepted in calls if attempt == 2
    } == {0, 1}
    assert [item.rejection_reason for item in receipt.variants].count("symmetry") == 1


def test_direct_selection_returns_five_distinct_variants_in_canonical_order():
    distinct = tuple(
        CampaignVariantV1.build({"candidate": index}) for index in range(5)
    )
    duplicate = distinct[0]

    accepted, receipts = select_campaign_variants_v1(
        tuple(
            (index, 1, variant)
            for index, variant in enumerate((*reversed(distinct), duplicate))
        ),
        result_count=5,
    )

    assert len(accepted) == 5
    assert [item.semantic_hash for item in accepted] == sorted(
        item.semantic_hash for item in distinct
    )
    assert [item.rejection_reason for item in receipts].count("duplicate") == 1


def _decision_state(*, seat_shift: int = 0, suit_shift: int = 0):
    seats = ("north", "west", "south", "east")
    hands = {
        seats[(seat + seat_shift) % 4]: [
            {"id": (((card // 8) + suit_shift) % 4) * 8 + card % 8} for card in cards
        ]
        for seat, cards in enumerate(
            ((0, 9, 18), (1, 10, 19), (2, 11, 20), (3, 12, 21))
        )
    }
    return {
        "contract": "sun",
        "trumpSuit": None,
        "currentPlayer": seats[seat_shift % 4],
        "buyer": seats[(1 + seat_shift) % 4],
        "dealer": seats[(3 + seat_shift) % 4],
        "hands": hands,
        "currentTrick": None,
    }


def test_rules_preserving_seat_and_suit_renaming_has_one_symmetry_key():
    original = build_rules_preserving_diversity_document_v1(_decision_state())
    renamed = build_rules_preserving_diversity_document_v1(
        _decision_state(seat_shift=1, suit_shift=2)
    )
    assert original == renamed
    assert decision_state_card_vector_v1(
        _decision_state()
    ) != decision_state_card_vector_v1(_decision_state(seat_shift=1, suit_shift=2))

    variants = tuple(
        CampaignVariantV1.build({"candidate": index}, symmetry_document=document)
        for index, document in enumerate((original, renamed))
    )
    accepted, receipts = select_campaign_variants_v1(
        tuple((index, 1, variant) for index, variant in enumerate(variants)),
        result_count=2,
        symmetry_key=rules_preserving_symmetry_key_v1,
    )
    assert len(accepted) == 1
    assert [item.rejection_reason for item in receipts].count("symmetry") == 1


def test_hokum_symmetry_keeps_trump_identity_while_renaming_suits():
    original_state = _decision_state()
    original_state.update({"contract": "hokum", "trumpSuit": "spades"})
    renamed_state = _decision_state(seat_shift=2, suit_shift=2)
    renamed_state.update({"contract": "hokum", "trumpSuit": "diamonds"})

    assert build_rules_preserving_diversity_document_v1(
        original_state
    ) == build_rules_preserving_diversity_document_v1(renamed_state)


def test_precontract_decision_uses_bidding_actor_and_visible_purchase_card():
    state = _decision_state()
    state["contract"] = None
    state["currentPlayer"] = None
    state["bidding"] = {"actor": "north"}
    state["purchaseCard"] = {"id": 31}
    document = build_rules_preserving_diversity_document_v1(state)
    assert document["contract"] == "none"
    assert 9 in document["decisionStateCardVector"]


def test_purchase_card_in_current_trick_uses_play_status_not_purchase_status():
    state = _decision_state()
    state["purchaseCard"] = {"id": 31}
    state["currentTrick"] = {
        "leader": "north",
        "plays": [{"seat": "north", "card": {"id": 31}}],
    }

    document = build_rules_preserving_diversity_document_v1(state)

    assert 4 in document["decisionStateCardVector"]
    assert 9 not in document["decisionStateCardVector"]


def _variant_with_vector(
    name: str,
    vector: list[int],
    *,
    principal_variation: list[str] | None = None,
) -> CampaignVariantV1:
    return CampaignVariantV1.build(
        {
            "candidate": name,
            "principalVariationActionIds": principal_variation or ["D7"],
        },
        symmetry_document={
            "schemaVersion": 1,
            "policyId": "rules_preserving_v1",
            "distancePolicyId": "decision_state_card_hamming_v1",
            "contract": "sun",
            "buyerOffset": 1,
            "dealerOffset": 3,
            "currentLeaderOffset": None,
            "decisionStateCardVector": vector,
            "currentTrickVector": [],
        },
    )


def test_minimum_decision_state_card_distance_rejects_near_variant():
    first = _variant_with_vector("first", [0] * 32)
    near = _variant_with_vector("near", [1, 1, *([0] * 30)])
    accepted, receipts = select_campaign_variants_v1(
        ((0, 1, first), (1, 1, near)),
        result_count=2,
        distance=decision_state_card_hamming_distance_v1,
        minimum_distance=4,
    )
    assert len(accepted) == 1
    rejected = next(item for item in receipts if not item.accepted)
    assert rejected.rejection_reason == "decision_state_distance"
    assert rejected.nearest_accepted_distance == 2


def test_semantic_distance_requires_decision_state_and_principal_variation() -> None:
    first = _variant_with_vector(
        "first",
        [0] * 32,
        principal_variation=["CA", "C7", "SK"],
    )
    same_line = _variant_with_vector(
        "same-line",
        [1, 1, 1, 1, *([0] * 28)],
        principal_variation=["CA", "C7", "SK"],
    )
    distinct = _variant_with_vector(
        "distinct",
        [2, 2, 2, 2, *([0] * 28)],
        principal_variation=["CA", "C7", "SQ"],
    )

    assert semantic_threshold_distance_v1(
        first,
        same_line,
        minimum_decision_state_distance=4,
        minimum_principal_variation_distance=1,
    ) == 1
    assert semantic_threshold_distance_v1(
        first,
        distinct,
        minimum_decision_state_distance=4,
        minimum_principal_variation_distance=1,
    ) == 2


def test_distance_governed_campaign_accepts_five_meaningful_variants():
    variants = tuple(
        _variant_with_vector(
            f"variant-{index}",
            [index if card < 4 else 0 for card in range(32)],
        )
        for index in range(5)
    )
    receipt = DeterministicCpSatCampaignCoordinatorV1(
        symmetry_key=rules_preserving_symmetry_key_v1,
        symmetry_policy_id="rules_preserving_v1",
        distance=decision_state_card_hamming_distance_v1,
        minimum_distance=4,
        distance_policy_id="decision_state_card_hamming_v1",
    ).run(
        shards=_shards(5),
        solver=lambda request: CampaignShardOutcomeV1(
            (variants[request.shard.shard_index],)
        ),
        budget=_budget(results=5),
    )
    assert receipt.status == "completed"
    assert len(receipt.accepted_variants) == 5
    assert receipt.minimum_distance == 4
    assert receipt.distance_policy_id == "decision_state_card_hamming_v1"


def test_timing_evidence_is_immutable_but_excluded_from_semantic_identity():
    shards = _shards(1)

    def run_with_timing(duration: int):
        variant = CampaignVariantV1.build(
            {"state": "fixed"},
            stage_timings=(CampaignStageTimingV1("tail", duration, duration),),
        )
        return DeterministicCpSatCampaignCoordinatorV1().run(
            shards=shards,
            solver=lambda _request: CampaignShardOutcomeV1((variant,)),
            budget=_budget(results=1, workers=1),
        )

    first = run_with_timing(1)
    second = run_with_timing(999)

    assert first.semantic_sha256 == second.semantic_sha256
    assert (
        first.accepted_variants[0].semantic_hash
        == second.accepted_variants[0].semantic_hash
    )
    assert (
        first.accepted_variants[0].stage_timings
        != second.accepted_variants[0].stage_timings
    )
    assert first.receipt_sha256 != second.receipt_sha256


def test_controlled_cancellation_preserves_completed_solver_evidence(campaign_clock, monkeypatch):
    shards = _shards(2)
    cancellation = threading.Event()
    release = threading.Event()
    workers_started = threading.Barrier(3)
    wait = coordinator.wait
    requests = []

    def wait_for_cancellation(futures, **kwargs):
        if not cancellation.is_set():
            workers_started.wait()
            cancellation.set()
            return set(), set(futures)
        assert all(request.cancellation.is_cancelled for request in requests)
        campaign_clock[0] = 40_000_000
        release.set()
        return wait(futures)

    monkeypatch.setattr(coordinator, "wait", wait_for_cancellation)

    def solver(request):
        requests.append(request)
        request.report_progress(
            {
                "currentStage": "tailConstruction",
                "modelHash": _hash("tail-model"),
                "variableCount": 32,
                "constraintCount": 64,
                "solverStatus": "running",
                "branches": 0,
                "conflicts": 0,
            }
        )
        workers_started.wait()
        request.cancellation.wait()
        release.wait()
        return CampaignShardOutcomeV1(
            (),
            (CampaignStageTimingV1("tailConstruction", 1_000_000),),
            {
                "currentStage": "tailConstruction",
                "modelHash": _hash("tail-model"),
                "variableCount": 32,
                "constraintCount": 64,
                "solverStatus": "unknown",
                "branches": 7,
                "conflicts": 3,
            },
        )

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=solver,
        budget=_budget(results=1, workers=2, wall=2.0),
        cancellation_event=cancellation,
    )

    assert receipt.status == "cancelled"
    assert receipt.semantic_sha256 is None
    assert {item.status for item in receipt.attempts} == {"cancelled"}
    attempt = receipt.attempts[0].to_dict()
    assert attempt["partialEvidence"]["currentStage"] == "tailConstruction"
    assert attempt["partialEvidence"]["variableCount"] == 32
    assert attempt["partialEvidence"]["branches"] == 7
    assert attempt["partialEvidence"]["conflicts"] == 3
    assert attempt["partialEvidence"]["cancellation"]["acknowledged"] is True
    assert attempt["partialEvidence"]["cancellation"]["latencyNs"] == 40_000_000
    assert {item["stage"] for item in attempt["stageTimings"]} >= {
        "tailConstruction",
        "shard_solver",
        "cancellation_acknowledgement",
    }


def test_cancellation_does_not_relabel_a_returned_variant_incomplete(campaign_clock):
    cancellation = threading.Event()
    live = []
    variant = _variant_with_vector("finished", [0, 1, 2, 3])
    def solver(request):
        request.report_progress({"currentStage": "playingQuality",
            "completedCandidates": 1, "candidateHash": _hash("finished")})
        cancellation.set()
        return CampaignShardOutcomeV1((variant,))
    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=_shards(1), solver=solver, budget=_budget(results=1, workers=1),
        cancellation_event=cancellation, progress_callback=live.append)
    assert any(variant.semantic_hash in attempt.variant_hashes for attempt in receipt.attempts)
    assert not any(item.get("candidateStatus") == "incomplete" for item in live)


def test_budget_exhaustion_preserves_completed_exact_variants(monkeypatch, campaign_clock):
    shards = _shards(2)
    wait = coordinator.wait

    def wait_then_expire(*args, **kwargs):
        done, pending = wait(*args, **kwargs)
        if done:
            campaign_clock[0] = 100_000_000
        return done, pending

    monkeypatch.setattr(coordinator, "wait", wait_then_expire)

    def solver(request):
        if request.shard.shard_index == 0:
            return CampaignShardOutcomeV1(
                (CampaignVariantV1.build({"candidate": "completed"}),)
            )
        request.cancellation.wait()
        return CampaignShardOutcomeV1(())

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=solver,
        budget=_budget(results=2, workers=2, wall=0.1),
    )

    assert receipt.status == "budget_exhausted"
    assert receipt.semantic_sha256 is None
    assert len(receipt.accepted_variants) == 1
    assert len(receipt.variants) == 1
    assert receipt.variants[0].accepted is True


@pytest.mark.parametrize("target_satisfied", [False, True])
def test_cancellation_joins_solver_only_when_target_is_satisfied(
    target_satisfied, campaign_clock, monkeypatch,
):
    extra = 1 if target_satisfied else 0
    started, release, finished = (threading.Event() for _ in range(3))
    requests = []
    workers = []
    cancellation = threading.Event()
    executor = coordinator.ThreadPoolExecutor
    wait = coordinator.wait

    class ObserveShutdown(executor):
        def shutdown(self, wait=True, **kwargs):
            if wait:
                release.set()
            super().shutdown(wait=wait, **kwargs)

    def wait_for_acknowledgement(futures, **kwargs):
        started.wait()
        request = requests[0]
        if request.cancellation.is_cancelled:
            request.cancellation_acknowledgement.wait()
            campaign_clock[0] = 225_000_000
            return set(), set(futures)
        if not target_satisfied:
            cancellation.set()
            return set(), set(futures)
        return wait(futures, return_when=kwargs["return_when"])

    monkeypatch.setattr(coordinator, "ThreadPoolExecutor", ObserveShutdown)
    monkeypatch.setattr(coordinator, "wait", wait_for_acknowledgement)

    def solver(request):
        if request.shard.shard_index != extra:
            started.wait()
            return CampaignShardOutcomeV1(
                (CampaignVariantV1.build({"candidate": "target"}),)
            )
        requests.append(request)
        workers.append(threading.current_thread())
        started.set()
        request.cancellation.wait()
        request.cancellation_requested()
        release.wait()
        finished.set()
        return CampaignShardOutcomeV1(())

    try:
        receipt = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=_shards(extra + 1),
            solver=solver,
            budget=_budget(results=1, workers=extra + 1),
            cancellation_event=cancellation,
            replenish_diversity_rejections=target_satisfied,
        )
        assert finished.is_set() is target_satisfied
        if target_satisfied:
            assert not workers[0].is_alive()
        assert receipt.status == ("completed" if target_satisfied else "cancelled")
        attempt = next(item for item in receipt.attempts if item.shard_index == extra)
        assert attempt.status == "cancelled"
        evidence = attempt.partial_evidence["cancellation"]
        assert evidence["acknowledged"] is True
        assert evidence["workerTerminated"] is target_satisfied
        assert evidence["latencyNs"] <= 225_000_000
    finally:
        release.set()
        if started.is_set():
            finished.wait()


def test_replenishing_campaign_uses_same_canonical_prefix_across_completion_orders(monkeypatch):
    shards = _shards(4)
    receipts = []
    for reverse in (False, True):
        calls: list[int] = []

        def solver(request):
            calls.append(request.shard.shard_index)
            return CampaignShardOutcomeV1(
                (CampaignVariantV1.build({"candidate": request.shard.shard_index}),)
            )

        with monkeypatch.context() as patch:
            _deliver_shards_in_order(patch, (3, 2, 1, 0) if reverse else (0, 1, 2, 3))
            receipt = DeterministicCpSatCampaignCoordinatorV1().run(
                shards=shards,
                solver=solver,
                budget=_budget(results=2, workers=4, wall=2.0),
                replenish_diversity_rejections=True,
            )
        assert receipt.status == "completed"
        assert receipt.semantic_sha256 is not None
        assert len(receipt.accepted_variants) == 2
        assert sorted(calls) == [0, 1, 2, 3]
        completed_shards = {
            item.shard_index
            for item in receipt.attempts
            if item.status == "completed"
        }
        assert len(completed_shards) == 2
        assert {item.status for item in receipt.attempts} <= {
            "completed",
            "cancelled",
            "not_required",
        }
        receipts.append(receipt)

    assert receipts[0].semantic_sha256 == receipts[1].semantic_sha256
    assert {
        item.shard_index
        for item in receipts[0].attempts
        if item.status == "completed"
    } == {
        item.shard_index
        for item in receipts[1].attempts
        if item.status == "completed"
    }
    assert [item.semantic_hash for item in receipts[0].accepted_variants] == [
        item.semantic_hash for item in receipts[1].accepted_variants
    ]


@pytest.mark.parametrize("acknowledge_at_ms,expected", [(40, True), (226, False)])
def test_acknowledgement_gate_is_not_confused_with_poll_interval(
    acknowledge_at_ms, expected, campaign_clock, monkeypatch,
):
    cancellation, ready, release, finished = (threading.Event() for _ in range(4))
    requests = []

    def solver(request):
        requests.append(request)
        ready.set()
        release.wait()
        finished.set()
        return CampaignShardOutcomeV1(())

    def advance_poll(futures, *, timeout, return_when):
        ready.wait()
        request = requests[0]
        if request.cancellation.is_cancelled:
            campaign_clock[0] += round(timeout * 1_000_000_000)
            if campaign_clock[0] >= acknowledge_at_ms * 1_000_000:
                request.cancellation_requested()
        else:
            cancellation.set()
        return set(), set(futures)

    monkeypatch.setattr(coordinator, "wait", advance_poll)
    try:
        receipt = DeterministicCpSatCampaignCoordinatorV1().run(
            shards=_shards(1), solver=solver, budget=_budget(results=1, workers=1),
            cancellation_event=cancellation,
        )
        assert receipt.status == "cancelled"
        attempt = receipt.attempts[0]
        assert attempt.status == ("cancelled" if expected else "cancellation_unacknowledged")
        evidence = attempt.partial_evidence["cancellation"]
        assert evidence["pollLimitMilliseconds"] == 5
        assert evidence["acknowledged"] is expected
        assert evidence["workerTerminated"] is False
        assert evidence["latencyNs"] == min(acknowledge_at_ms, 225) * 1_000_000
    finally:
        release.set()
        if ready.is_set():
            finished.wait()


def test_incomplete_proof_is_retried_once_and_streams_attempt_progress():
    shards = _shards(1, attempts=3)
    progress: list[dict] = []

    def solver(request):
        request.report_progress({"currentStage": "proof"})
        if request.attempt == 1:
            return CampaignShardOutcomeV1(
                (), partial_evidence={"proofRetryEligible": True}
            )
        return CampaignShardOutcomeV1(
            (CampaignVariantV1.build({"seed": request.shard.seed_start}),)
        )

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=solver,
        budget=_budget(results=1, workers=1),
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert receipt.status == "completed"
    assert [(item.attempt, item.status) for item in receipt.attempts] == [
        (1, "completed"),
        (2, "completed"),
    ]
    assert any(item["currentStage"] == "proof_retry_scheduled" for item in progress)
    assert {item["attempt"] for item in progress} == {1, 2}
    assert all(item["shardId"] == shards[0].shard_id for item in progress)


def test_incomplete_search_uses_remaining_campaign_attempt():
    shards = _shards(1, attempts=2)
    progress: list[dict] = []

    def solver(request):
        if request.attempt == 1:
            return CampaignShardOutcomeV1(
                (), partial_evidence={"attemptRetryEligible": True}
            )
        return CampaignShardOutcomeV1(
            (CampaignVariantV1.build({"seed": request.shard.seed_start + 1}),)
        )

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=solver,
        budget=_budget(results=1, workers=1),
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert receipt.status == "completed"
    assert [item.attempt for item in receipt.attempts] == [1, 2]
    assert any(
        item["currentStage"] == "attempt_retry_scheduled" for item in progress
    )


def test_diversity_campaign_does_not_retry_before_current_wave_reaches_target():
    shards = _shards(2, attempts=2)
    calls: list[tuple[int, int]] = []

    def solver(request):
        calls.append((request.shard.shard_index, request.attempt))
        return CampaignShardOutcomeV1(
            (CampaignVariantV1.build({"shard": request.shard.shard_index}),),
            partial_evidence={"attemptRetryEligible": True},
        )

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=solver,
        budget=_budget(results=2, workers=2),
        replenish_diversity_rejections=True,
    )

    assert receipt.status == "completed"
    assert sorted(calls) == [(0, 1), (1, 1)]


def test_diversity_rejection_streams_before_campaign_completion():
    shards = _shards(2)
    progress: list[dict] = []
    duplicate = CampaignVariantV1.build({"seed": 1})

    receipt = DeterministicCpSatCampaignCoordinatorV1().run(
        shards=shards,
        solver=lambda _request: CampaignShardOutcomeV1((duplicate,)),
        budget=_budget(results=2, workers=1),
        progress_callback=lambda value: progress.append(dict(value)),
    )

    assert receipt.status == "not_found_within_budget"
    rejected = [
        item
        for item in progress
        if item["currentStage"] == "variant_selection_terminal"
    ]
    assert len(rejected) == 1
    assert rejected[0]["candidateStatus"] == "duplicate"
    assert rejected[0]["candidateHash"] == duplicate.semantic_hash


def test_invalid_worker_declaration_and_overlapping_ranges_fail_closed():
    try:
        CampaignBudgetV1(
            result_count=1,
            shard_worker_count=1,
            maximum_wall_time_seconds=1,
            cp_sat_workers_per_solve=2,
        )
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("multi-worker CP-SAT declaration was accepted")

    first = _shards(1)[0]
    overlapping = type(first).build(
        run_code=first.run_code,
        run_hash=first.run_hash,
        recipe_hash=first.recipe_hash,
        shard_index=1,
        seed_start=first.seed_start,
        seed_count=1,
    )
    try:
        DeterministicCpSatCampaignCoordinatorV1().run(
            shards=(first, overlapping),
            solver=lambda _request: CampaignShardOutcomeV1(()),
            budget=_budget(results=1),
        )
    except ValueError as error:
        assert "must not overlap" in str(error)
    else:
        raise AssertionError("overlapping seed ranges were accepted")

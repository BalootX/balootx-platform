from __future__ import annotations

import copy
import concurrent.futures
import multiprocessing
import time
from dataclasses import replace

import pytest
import yaml
from deterministic_work import HANG_GUARD

import baloot_platform.challenge.analysis_pipeline as pipeline_module
from baloot_platform.agents.profiles import AgentProfileRegistry
from baloot_platform.analysis.decision_scores import (
    PrimaryDecisionScoreEvidenceV1,
    bid_score_rows_v1,
    canonical_sha256_v1,
    decision_score_rows_hash_v1,
    legal_action_set_hash_v1,
    play_score_rows_v1,
)
from baloot_platform.challenge.analysis_pipeline import (
    CascadedTraceAnalysisProfileV1,
    CascadedTraceAnalyzerV1,
    DecisionAnalysisExecutorProfileV1,
    StagedTraceAnalyzerV1,
    TraceAnalysisResultV1,
    TraceDecisionAnalysisV3,
    TraceAnalysisProfileV1,
    _abort_process_executor,
    _canonical_analysis_results,
    _canonical_worker_failure,
    _decision_seed,
)
from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1
from baloot_platform.challenge.finalist_adjudication import (
    AdaptiveFinalistTraceAnalyzerV1,
)
from baloot_platform.challenge.finalist_competence import (
    build_finalist_adjudication_profile_v2,
    canonical_finalist_cohort_v1,
    canonical_score_support_classification_v1,
    finalist_adjudication_target_required_v2,
    project_finalist_adjudication_v2,
)
from baloot_platform.challenge.replay import TraceDecisionProjectionStoreV1
from baloot_platform.challenge.profiles import SelectionProfileV1
from baloot_platform.challenge.simulation import NaturalLaffaSimulatorV1
from baloot_platform.config import build_bundle
from baloot_platform.core.rules import RuleBook


def _non_cooperative_worker(ready, seconds: float) -> None:
    with ready:
        ready.send("started")
    time.sleep(seconds)


@pytest.fixture(scope="module")
def trace_bundle(root):
    bundle = build_bundle(root)
    registry = AgentProfileRegistry.from_bundle(root)
    simulator = NaturalLaffaSimulatorV1(
        RuleBook.load(root / "configs/rules/saudi_common_v1.yaml"),
        yaml.safe_load(
            (root / "configs/conventions/conservative_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
        bundle_hash=bundle.sha256,
        configuration_snapshot=bundle.configuration_snapshot(),
        agent_profile=registry.get("strong"),
        maximum_deal_attempts=4,
    )
    return simulator.simulate(73031).final_trace, bundle


def _analyzer(
    bundle,
    *,
    workers: int,
    timeout: float = 900.0,
    cohort_authority_hash: str | None = None,
):
    return StagedTraceAnalyzerV1(
        bundle.rules,
        bundle.convention,
        TraceAnalysisProfileV1.build(
            bid_samples=2,
            play_samples=2,
            exact_allocation_limit=50,
            exact_search_max_remaining=2,
            maximum_sampling_attempts=2_000,
            exact_search_max_nodes=10_000,
            exact_search_max_seconds=0.1,
            exact_search_max_cache_entries=5_000,
        ),
        match_value_profile=bundle.component("match_value"),
        executor_profile=DecisionAnalysisExecutorProfileV1.build(
            worker_count=workers,
            maximum_pending_per_worker=1,
            maximum_seconds=timeout,
            poll_seconds=0.01,
        ),
        cohort_authority_hash=cohort_authority_hash,
    )


def test_executor_profile_is_governed_default_safe_and_bounded():
    assert StagedTraceAnalyzerV1.descriptor.version == "1.3.0"
    assert CascadedTraceAnalyzerV1.descriptor.version == "1.3.0"
    assert {
        "canonical_decision_score_evidence",
        "forced_legality_proof",
    }.issubset(StagedTraceAnalyzerV1.descriptor.capabilities)
    first = DecisionAnalysisExecutorProfileV1.build()
    second = DecisionAnalysisExecutorProfileV1.build()
    assert first == second
    assert first.worker_count == 1
    assert first.maximum_pending_per_worker == 1
    assert first.to_dict()["sha256"] == first.sha256
    with pytest.raises(ValueError, match="worker count"):
        DecisionAnalysisExecutorProfileV1.build(worker_count=5)
    with pytest.raises(ValueError, match="pending tasks"):
        DecisionAnalysisExecutorProfileV1.build(maximum_pending_per_worker=3)


def test_analyzer_accepts_only_an_explicit_canonical_cohort_authority(trace_bundle):
    _trace, bundle = trace_bundle
    explicit = "f" * 64

    assert (
        _analyzer(
            bundle,
            workers=1,
            cohort_authority_hash=explicit,
        ).cohort_authority_hash
        == explicit
    )
    with pytest.raises(ValueError, match="cohort authority"):
        _analyzer(
            bundle,
            workers=1,
            cohort_authority_hash="not-a-canonical-sha256",
        )


def test_one_and_two_workers_have_exact_normalized_evidence_parity(trace_bundle):
    trace, bundle = trace_bundle
    authority = "e" * 64
    serial = _analyzer(
        bundle,
        workers=1,
        cohort_authority_hash=authority,
    ).analyze_trace(trace)
    parallel = _analyzer(
        bundle,
        workers=2,
        cohort_authority_hash=authority,
    ).analyze_trace(trace)

    assert parallel.sha256 == serial.sha256
    assert parallel.to_dict() == serial.to_dict()
    assert tuple(item.sha256 for item in parallel.decisions) == tuple(
        item.sha256 for item in serial.decisions
    )
    assert all(
        item.primary_decision_score_evidence.analysis_identity_hash == authority
        for item in serial.decisions
        if not item.forced
        and item.primary_decision_score_evidence is not None
    )
    assert TraceAnalysisResultV1.from_dict(
        serial.to_dict(),
        ledger=trace.decision_ledger,
    ) == serial


def test_bid_and_play_audit_evidence_preserves_exact_analyzer_results(
    trace_bundle,
    monkeypatch,
):
    trace, bundle = trace_bundle
    authority = "d" * 64
    analyzer = _analyzer(
        bundle,
        workers=1,
        cohort_authority_hash=authority,
    )
    snapshots = TraceDecisionProjectionStoreV1(trace, bundle.rules).replay.decisions
    selected = tuple(
        next(
            snapshot
            for snapshot in snapshots
            if not snapshot.decision.forced
            and snapshot.decision.decision_kind == kind
        )
        for kind in ("bid", "play")
    )
    captured: dict[str, object] = {}
    original_analyze_bid = pipeline_module.analyze_bid
    original_analyze_play = pipeline_module.analyze

    def capture_bid(*args, **kwargs):
        result = original_analyze_bid(*args, **kwargs)
        captured["bid"] = result
        return result

    def capture_play(*args, **kwargs):
        result = original_analyze_play(*args, **kwargs)
        captured["play"] = result
        return result

    monkeypatch.setattr(pipeline_module, "analyze_bid", capture_bid)
    monkeypatch.setattr(pipeline_module, "analyze", capture_play)
    analyses = analyzer.analyze_decisions(
        trace,
        tuple(snapshot.decision.decision_id for snapshot in selected),
    )
    records = {item.decision_id: item for item in trace.decision_ledger.records}

    assert {item.decision_kind for item in analyses} == {"bid", "play"}
    for item in analyses:
        record = records[item.decision_id]
        evidence = item.primary_decision_score_evidence
        assert evidence is not None
        assert item.forced_legality_proof is None
        assert evidence.actor == record.acting_seat.name.lower()
        assert evidence.public_prefix_hash == record.public_prefix_hash
        assert evidence.legal_action_set_hash == legal_action_set_hash_v1(
            record.legal_actions
        )
        assert evidence.rules_hash == bundle.rules.sha256
        assert evidence.analysis_identity_hash == authority
        assert evidence.trace_analysis_profile_hash == analyzer.profile.sha256
        assert evidence.cohort_seed_policy == "actor_observer_cohort_seed_v1"
        assert evidence.cohort_index == 0
        assert evidence.cohort_seed == _decision_seed(
            record,
            authority,
            seed_namespace="",
            seed_offset=0,
        )
        assert evidence.backend == analyzer.profile.search_backend_id
        assert evidence.method == item.method
        assert evidence.hidden_deal_support == item.hidden_deals
        assert evidence.effective_sample_size == item.effective_sample_size
        assert evidence.recommended_action == item.best_action
        assert TraceDecisionAnalysisV3.from_dict(item.to_dict()) == item

        if item.decision_kind == "bid":
            result = captured["bid"]
            assert evidence.objective == "expected_round_margin"
            assert evidence.tie_break == (
                "round_lead_probability_then_best_share_then_legal_action_order_v1"
            )
            assert evidence.rows == bid_score_rows_v1(result.choices)
        else:
            result = captured["play"]
            assert evidence.objective == analyzer.profile.objective
            assert evidence.tie_break == (
                "configured_objective_then_match_win_probability_then_expected_"
                "round_margin_then_contract_success_probability_then_round_lead_"
                "probability_then_best_share_then_card_id_v1"
            )
            assert evidence.rows == play_score_rows_v1(result.actions)


def test_score_evidence_mutation_and_binding_drift_fail_closed(trace_bundle):
    trace, bundle = trace_bundle
    analyzer = _analyzer(
        bundle,
        workers=1,
        cohort_authority_hash="c" * 64,
    )
    record = next(item for item in trace.decision_ledger.records if not item.forced)
    item = analyzer.analyze_decisions(trace, (record.decision_id,))[0]
    serialized = item.to_dict()
    serialized["primaryDecisionScoreEvidence"]["rows"][0][
        "expectedRoundMargin"
    ] += 1.0
    with pytest.raises(ValueError, match="checksum"):
        TraceDecisionAnalysisV3.from_dict(serialized)

    evidence = item.primary_decision_score_evidence
    assert evidence is not None
    rebound = copy.deepcopy(evidence.to_dict())
    rebound["publicPrefixHash"] = (
        ("0" if record.public_prefix_hash[0] != "0" else "1")
        + record.public_prefix_hash[1:]
    )
    rebound_core = {key: value for key, value in rebound.items() if key != "sha256"}
    rebound["sha256"] = canonical_sha256_v1(rebound_core)
    self_valid_rebound = PrimaryDecisionScoreEvidenceV1.from_mapping(rebound)
    with pytest.raises(ValueError, match="binding mismatch"):
        replace(
            item,
            primary_decision_score_evidence=self_valid_rebound,
        ).validate_ledger_binding(
            record,
            trace_analysis_profile_hash=analyzer.profile.sha256,
            rules_hash=bundle.rules.sha256,
        )

    missing = item.to_dict()
    missing["primaryDecisionScoreEvidence"] = None
    with pytest.raises(ValueError, match="requires primary decision score evidence"):
        TraceDecisionAnalysisV3.from_dict(missing)


def test_forced_decision_retains_only_explicit_singleton_legality_proof(
    trace_bundle,
):
    trace, bundle = trace_bundle
    analyzer = _analyzer(bundle, workers=1)
    record = next(item for item in trace.decision_ledger.records if item.forced)
    item = analyzer.analyze_decisions(trace, (record.decision_id,))[0]

    assert item.primary_decision_score_evidence is None
    proof = dict(item.forced_legality_proof or {})
    assert proof["proofType"] == "authoritative_forced_legal_action_v1"
    assert proof["legalActionCount"] == 1
    assert proof["legalActionSetHash"] == legal_action_set_hash_v1(
        record.legal_actions
    )
    assert proof["selectedAction"] == record.chosen_action
    assert proof["rulesHash"] == bundle.rules.sha256
    assert TraceDecisionAnalysisV3.from_dict(item.to_dict()) == item

    missing = item.to_dict()
    missing["forcedLegalityProof"] = None
    with pytest.raises(ValueError, match="requires an explicit forced legality proof"):
        TraceDecisionAnalysisV3.from_dict(missing)

    rebound = item.to_dict()
    rebound_proof = rebound["forcedLegalityProof"]
    rebound_proof["legalActionSetHash"] = "0" * 64
    rebound_core = {
        key: value for key, value in rebound_proof.items() if key != "sha256"
    }
    rebound_proof["sha256"] = canonical_sha256_v1(rebound_core)
    with pytest.raises(ValueError, match="singleton action-set checksum mismatch"):
        TraceDecisionAnalysisV3.from_dict(rebound)


def test_secondary_cohort_has_exact_serial_parallel_parity(trace_bundle):
    trace, bundle = trace_bundle
    decision_ids = tuple(
        item.decision_id
        for item in trace.decision_ledger.records
        if not item.forced
    )[:3]
    assert len(decision_ids) >= 2

    serial = _analyzer(bundle, workers=1).analyze_decisions(
        trace,
        decision_ids,
        seed_namespace="confirmation:v1",
        seed_offset=104_729,
    )
    parallel = _analyzer(bundle, workers=2).analyze_decisions(
        trace,
        decision_ids,
        seed_namespace="confirmation:v1",
        seed_offset=104_729,
    )

    assert tuple(item.to_dict() for item in parallel) == tuple(
        item.to_dict() for item in serial
    )
    assert tuple(item.sha256 for item in parallel) == tuple(
        item.sha256 for item in serial
    )


def test_primary_seed_ignores_trace_and_selected_result_identity(
    trace_bundle,
    monkeypatch,
):
    trace, bundle = trace_bundle
    analyzer = _analyzer(bundle, workers=1)
    snapshot = next(
        item
        for item in TraceDecisionProjectionStoreV1(
            trace,
            bundle.rules,
        ).replay.decisions
        if not item.decision.forced
    )
    observed_seeds: list[int] = []

    def capture_seed(_snapshot, seed, _cancel_check, *, cohort_index):
        assert cohort_index == 0
        observed_seeds.append(int(seed))
        return object()

    monkeypatch.setattr(analyzer, "_bid", capture_seed)
    monkeypatch.setattr(analyzer, "_play", capture_seed)
    analyzer._analyze_snapshots_serial(
        trace,
        (snapshot,),
        seed_namespace="",
        seed_offset=0,
        cancel_check=None,
        progress_callback=None,
    )

    mutated_record = replace(
        snapshot.decision,
        decision_id="DEC-RESULT-IDENTITY-MUST-NOT-SEED",
        chosen_action={"ignoredResultIdentity": True},
    )
    changed_trace_hash = (
        ("0" if trace.sha256[0] != "0" else "1") + trace.sha256[1:]
    )
    analyzer._analyze_snapshots_serial(
        replace(trace, sha256=changed_trace_hash),
        (replace(snapshot, decision=mutated_record),),
        seed_namespace="",
        seed_offset=0,
        cancel_check=None,
        progress_callback=None,
    )

    assert analyzer.cohort_authority_hash == analyzer.profile.sha256
    assert observed_seeds == [observed_seeds[0], observed_seeds[0]]


def test_actor_state_kind_authority_and_secondary_cohort_change_seed(trace_bundle):
    trace, bundle = trace_bundle
    analyzer = _analyzer(bundle, workers=1)
    record = next(
        item for item in trace.decision_ledger.records if not item.forced
    )
    authority = analyzer.cohort_authority_hash
    primary = _decision_seed(
        record,
        authority,
        seed_namespace="",
        seed_offset=0,
    )
    other_public_hash = (
        ("0" if record.public_prefix_hash[0] != "0" else "1")
        + record.public_prefix_hash[1:]
    )
    other_actor = next(
        seat for seat in type(record.acting_seat) if seat != record.acting_seat
    )
    other_kind = "play" if record.decision_kind == "bid" else "bid"
    other_authority = (
        ("0" if authority[0] != "0" else "1") + authority[1:]
    )

    assert _decision_seed(
        replace(record, public_prefix_hash=other_public_hash),
        authority,
        seed_namespace="",
        seed_offset=0,
    ) != primary
    assert _decision_seed(
        replace(record, acting_seat=other_actor),
        authority,
        seed_namespace="",
        seed_offset=0,
    ) != primary
    assert _decision_seed(
        replace(record, decision_kind=other_kind),
        authority,
        seed_namespace="",
        seed_offset=0,
    ) != primary
    assert _decision_seed(
        record,
        other_authority,
        seed_namespace="",
        seed_offset=0,
    ) != primary

    secondary = _decision_seed(
        record,
        authority,
        seed_namespace="confirmation:v1",
        seed_offset=104_729,
    )
    assert secondary != primary
    assert secondary == _decision_seed(
        record,
        authority,
        seed_namespace="confirmation:v1",
        seed_offset=104_729,
    )
    assert secondary != _decision_seed(
        record,
        authority,
        seed_namespace="confirmation:v1:cohort3",
        seed_offset=209_458,
    )


@pytest.mark.parametrize("decision_kind", ("bid", "play"))
def test_tampered_public_prefix_fails_before_sampling(
    trace_bundle,
    monkeypatch,
    decision_kind,
):
    trace, bundle = trace_bundle
    records = list(trace.decision_ledger.records)
    index = next(
        index
        for index, item in enumerate(records)
        if item.decision_kind == decision_kind
    )
    record = records[index]
    tampered_hash = "0" * 64 if record.public_prefix_hash != "0" * 64 else "1" * 64
    records[index] = replace(record, public_prefix_hash=tampered_hash)
    tampered_trace = replace(
        trace,
        decision_ledger=DecisionLedgerV1.build(trace.game_hash, records),
    )

    def sampling_must_not_start(*_args, **_kwargs):
        raise AssertionError("sampling started before public-prefix verification")

    monkeypatch.setattr(
        "baloot_platform.challenge.analysis_pipeline.analyze_bid",
        sampling_must_not_start,
    )
    monkeypatch.setattr(
        "baloot_platform.challenge.analysis_pipeline.analyze",
        sampling_must_not_start,
    )

    with pytest.raises(
        ValueError,
        match=rf"{decision_kind} actor-observer public-prefix hash mismatch",
    ):
        _analyzer(bundle, workers=1).analyze_trace(tampered_trace)


def test_canonical_reduction_and_failure_are_completion_order_independent(trace_bundle):
    trace, bundle = trace_bundle
    decisions = _analyzer(bundle, workers=1).analyze_decisions(
        trace,
        tuple(item.decision_id for item in trace.decision_ledger.records[:3]),
    )
    reversed_completion = {
        2: decisions[2],
        0: decisions[0],
        1: decisions[1],
    }
    assert _canonical_analysis_results(3, reversed_completion) == decisions

    first = ValueError("first ordinal failure")
    later = RuntimeError("later ordinal failure")
    assert _canonical_worker_failure({2: later, 1: first}) is first


@pytest.mark.real_clock
def test_parallel_analysis_honors_parent_cancellation(trace_bundle):
    trace, bundle = trace_bundle
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="trace analysis cancelled"):
        _analyzer(bundle, workers=2).analyze_trace(
            trace,
            cancel_check=lambda: time.monotonic() - started >= 0.05,
        )
    assert time.monotonic() - started < 10.0


@pytest.mark.real_clock
def test_parallel_analysis_enforces_governed_timeout(trace_bundle):
    trace, bundle = trace_bundle
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="governed timeout"):
        _analyzer(bundle, workers=2, timeout=0.05).analyze_trace(trace)
    assert time.monotonic() - started < 10.0


@pytest.mark.real_clock
def test_abort_process_executor_uses_terminate_then_kill_fallback():
    class FakeFuture:
        cancelled = False

        def cancel(self):
            self.cancelled = True
            return False

    class FakeProcess:
        alive = True
        terminated = False
        killed = False
        join_calls = 0

        def is_alive(self):
            return self.alive

        def join(self, *, timeout):
            assert timeout >= 0
            self.join_calls += 1

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.alive = False

    class FakeManagerThread:
        joined = False

        def join(self, *, timeout):
            assert timeout > 0
            self.joined = True

        def is_alive(self):
            return False

    class FakeExecutor:
        shutdown_call = None

        def __init__(self, process, manager_thread):
            self._processes = {process: process}
            self._executor_manager_thread = manager_thread

        def shutdown(self, *, wait, cancel_futures):
            self.shutdown_call = (wait, cancel_futures)

    future = FakeFuture()
    process = FakeProcess()
    manager_thread = FakeManagerThread()
    executor = FakeExecutor(process, manager_thread)

    _abort_process_executor(
        executor,
        (future,),
        maximum_wait_seconds=0.01,
    )

    assert future.cancelled
    assert executor.shutdown_call == (False, True)
    assert process.terminated
    assert process.killed
    assert process.join_calls == 3
    assert manager_thread.joined


@pytest.mark.real_clock
def test_abort_process_executor_is_bounded_owned_and_repeatable():
    context = multiprocessing.get_context("spawn")
    unrelated_reader, unrelated_writer = context.Pipe(duplex=False)
    unrelated = context.Process(
        target=_non_cooperative_worker,
        args=(unrelated_writer, 30.0),
    )
    unrelated.start()
    unrelated_writer.close()
    try:
        assert unrelated_reader.poll(HANG_GUARD), "unrelated test process did not start"
        assert unrelated_reader.recv() == "started"

        for _ in range(2):
            owned_reader, owned_writer = context.Pipe(duplex=False)
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
            )
            owned_processes = ()
            try:
                future = executor.submit(_non_cooperative_worker, owned_writer, 30.0)
                owned_processes = tuple(dict(executor._processes or {}).values())
                assert owned_reader.poll(HANG_GUARD), "owned test worker did not start"
                assert owned_reader.recv() == "started"

                started = time.monotonic()
                _abort_process_executor(
                    executor,
                    (future,),
                    maximum_wait_seconds=0.5,
                )

                assert time.monotonic() - started < 3.0
                assert owned_processes
                assert all(not process.is_alive() for process in owned_processes)
                assert unrelated.is_alive()
            finally:
                owned_reader.close()
                owned_writer.close()
                executor.shutdown(wait=False, cancel_futures=True)
                for process in owned_processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
    finally:
        unrelated_reader.close()
        if unrelated.is_alive():
            unrelated.terminate()
        unrelated.join(timeout=2.0)
        if unrelated.is_alive():
            unrelated.kill()
            unrelated.join(timeout=2.0)


def _receipt_reanalysis(
    trace,
    bundle,
    result: TraceAnalysisResultV1,
) -> TraceDecisionAnalysisV3:
    candidates = tuple(
        item
        for item in result.decisions
        if not item.forced and item.primary_decision_score_evidence is not None
    )
    primary = next(
        (
            item
            for item in candidates
            if canonical_sha256_v1(dict(item.actual_action))
            == canonical_sha256_v1(dict(item.best_action))
        ),
        candidates[0],
    )
    return _analyzer(
        bundle,
        workers=1,
        cohort_authority_hash="e" * 64,
    ).analyze_decisions(
        trace,
        (primary.decision_id,),
        seed_namespace="finalist_surrounding_adjudication_v2",
        seed_offset=104_729,
    )[0]


def test_adjudication_writer_rejects_ungoverned_secondary_and_third_seeds(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    reanalysis = _receipt_reanalysis(trace, bundle, result)
    AdaptiveFinalistTraceAnalyzerV1._validate_governed_cohort_seeds(
        trace,
        (reanalysis,),
        cohort_index=1,
        seed_namespace="finalist_surrounding_adjudication_v2",
        seed_offset=104_729,
    )

    third_value = copy.deepcopy(reanalysis.to_dict())
    third_score = third_value["primaryDecisionScoreEvidence"]
    record = next(
        item
        for item in trace.decision_ledger.records
        if item.decision_id == reanalysis.decision_id
    )
    third_score["cohortSeed"] = _decision_seed(
        record,
        third_score["analysisIdentityHash"],
        seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
        seed_offset=209_458,
    )
    _rebind_serialized_decision_analysis(third_value)
    third = TraceDecisionAnalysisV3.from_dict(third_value)
    AdaptiveFinalistTraceAnalyzerV1._validate_governed_cohort_seeds(
        trace,
        (third,),
        cohort_index=1,
        seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
        seed_offset=209_458,
    )

    drifted_value = copy.deepcopy(third.to_dict())
    drifted_value["primaryDecisionScoreEvidence"]["cohortSeed"] += 1
    _rebind_serialized_decision_analysis(drifted_value)
    drifted = TraceDecisionAnalysisV3.from_dict(drifted_value)
    with pytest.raises(ValueError, match="seed is not governed by the ledger"):
        AdaptiveFinalistTraceAnalyzerV1._validate_governed_cohort_seeds(
            trace,
            (drifted,),
            cohort_index=1,
            seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
            seed_offset=209_458,
        )


def _strict_schema3_receipt(
    result: TraceAnalysisResultV1,
    reanalysis: TraceDecisionAnalysisV3,
    *,
    ledger: DecisionLedgerV1,
) -> dict[str, object]:
    candidates = tuple(
        item
        for item in result.decisions
        if not item.forced and item.primary_decision_score_evidence is not None
    )
    source_primary = next(
        (
            item
            for item in candidates
            if canonical_sha256_v1(dict(item.actual_action))
            == canonical_sha256_v1(dict(item.best_action))
        ),
        candidates[0],
    )
    assert reanalysis.decision_id == source_primary.decision_id
    source_reanalysis_score = reanalysis.primary_decision_score_evidence
    assert source_reanalysis_score is not None
    primary_profile_core = {
        "schemaVersion": 1,
        "screeningProfileHash": result.profile_hash,
        "confirmationProfileHash": result.profile_hash,
        "maximumConfirmationDecisionsPerTrace": 64,
        "minimumScreeningScore": 0.0,
    }
    primary_profile = CascadedTraceAnalysisProfileV1.from_mapping({
        **primary_profile_core,
        "sha256": canonical_sha256_v1(primary_profile_core),
    })
    profile = build_finalist_adjudication_profile_v2(
        {
            "seed_offset": 104_729,
            "minimum_hidden_deals": 2,
            "minimum_best_share": 0.0,
            "maximum_competence_expected_margin_regret": 1_000_000.0,
            "maximum_competence_match_win_drop": 1.0,
            "maximum_expected_margin_drift": 1_000_000.0,
            "maximum_cohorts": 3,
            "minimum_agreement": 2,
            "third_cohort_on_disagreement": True,
            "require_same_recommended_action": True,
        },
        primary_profile_hash=primary_profile.sha256,
        reanalysis_profile_hash=(
            source_reanalysis_score.trace_analysis_profile_hash
        ),
        maximum_decisions_per_trace=64,
        minimum_confidence=0.0,
        minimum_quality=0.0,
    )
    selection = SelectionProfileV1.build({
        "schemaVersion": 1,
        "id": "closed-receipt-selection",
        "revision": "1",
        "weights": {"outcomeImpact": 1.0},
        "allowedTruthLabels": ["exact_optimal", "robust_best_known"],
        "minimumConfidence": 0.0,
        "minimumScore": 0.0,
        "requireVerifiedTacticalPlay": False,
        "minimumSurroundingQuality": 0.0,
        "minimumSurroundingConfidence": 0.0,
        "minimumSurroundingMateriality": 0.0,
        "maximumSurroundingMaterialMistakes": 0,
        "maximumSurroundingUnverifiedDecisions": 0,
        "maximumSurroundingCatastrophicCrossings": 0,
    })
    targets = tuple(
        item
        for item in sorted(candidates, key=lambda value: (value.ordinal, value.decision_id))
        if finalist_adjudication_target_required_v2(
            item,
            adjudication_profile=profile,
            selection_profile=selection.to_dict(),
        )
    )
    assert targets
    records = {item.decision_id: item for item in ledger.records}

    def governed_reanalysis(
        primary: TraceDecisionAnalysisV3,
        *,
        namespace: str,
        offset: int,
    ) -> TraceDecisionAnalysisV3:
        value = copy.deepcopy(primary.to_dict())
        score = value["primaryDecisionScoreEvidence"]
        score.update({
            "analysisIdentityHash": (
                source_reanalysis_score.analysis_identity_hash
            ),
            "traceAnalysisProfileHash": (
                source_reanalysis_score.trace_analysis_profile_hash
            ),
            "cohortSeedPolicy": source_reanalysis_score.cohort_seed_policy,
            "cohortIndex": 1,
            "cohortSeed": _decision_seed(
                records[primary.decision_id],
                source_reanalysis_score.analysis_identity_hash,
                seed_namespace=namespace,
                seed_offset=offset,
            ),
        })
        _rebind_serialized_decision_analysis(value)
        return TraceDecisionAnalysisV3.from_dict(value)

    evaluations: list[dict[str, object]] = []
    third_ids: list[str] = []
    for primary in targets:
        secondary = governed_reanalysis(
            primary,
            namespace="finalist_surrounding_adjudication_v2",
            offset=104_729,
        )
        pair = _canonical_adjudication_evaluation(
            primary,
            (primary, secondary),
            profile,
        )
        if pair["strictReasons"]:
            third = governed_reanalysis(
                primary,
                namespace="finalist_surrounding_adjudication_v2:cohort3",
                offset=209_458,
            )
            evaluation = _canonical_adjudication_evaluation(
                primary,
                (primary, secondary, third),
                profile,
            )
            third_ids.append(primary.decision_id)
        else:
            evaluation = pair
        evaluations.append(evaluation)
    target_ids = [item.decision_id for item in targets]
    unresolved_ids = sorted(
        item["decisionId"]
        for item in evaluations
        if item["status"] == "unresolved"
    )
    competent_ids = sorted(
        item["decisionId"]
        for item in evaluations
        if item["status"] == "primary_robust_competent"
    )
    adjudication_core = {
        "schemaVersion": 2,
        "verdictSchemaVersion": 2,
        "profileHash": profile["sha256"],
        "adjudicationProfile": profile,
        "selectionProfile": selection.to_dict(),
        "primaryProfile": primary_profile.to_dict(),
        "primaryProfileHash": profile["primaryProfileHash"],
        "reanalysisProfileHash": profile["reanalysisProfileHash"],
        "seedNamespace": "finalist_surrounding_adjudication_v2",
        "seedOffset": 104_729,
        "thirdSeedNamespace": "finalist_surrounding_adjudication_v2:cohort3",
        "thirdSeedOffset": 209_458,
        "targetDecisionIds": target_ids,
        "boundedDecisionIds": target_ids,
        "overflowDecisionIds": [],
        "thirdCohortDecisionIds": third_ids,
        "evaluations": evaluations,
        "unresolvedDecisionIds": unresolved_ids,
        "competentDecisionIds": competent_ids,
    }
    effective_profile_hash = canonical_sha256_v1({
        "schemaVersion": 2,
        "primaryProfileHash": profile["primaryProfileHash"],
        "adjudicationProfileHash": profile["sha256"],
        "selectionProfileHash": selection.sha256,
    })
    receipt = copy.deepcopy(result.to_dict())
    receipt.update({
        "schemaVersion": 3,
        "profileHash": effective_profile_hash,
        "analyzerId": "adaptive_finalist_trace_analyzer_v1",
        "analyzerVersion": "1.3.0",
        "confirmedDecisionIds": [],
        "screeningProfileHash": result.profile_hash,
        "confirmationProfileHash": result.profile_hash,
        "adjudicationEvidence": {
            **adjudication_core,
            "sha256": canonical_sha256_v1(adjudication_core),
        },
        "unresolvedDecisionIds": adjudication_core["unresolvedDecisionIds"],
    })
    _rebind_schema3_receipt(receipt)
    return receipt


def _canonical_adjudication_evaluation(
    primary: TraceDecisionAnalysisV3,
    cohorts: tuple[TraceDecisionAnalysisV3, ...],
    profile: dict[str, object],
) -> dict[str, object]:
    classifications = tuple(
        canonical_score_support_classification_v1(
            item.primary_decision_score_evidence
        )
        for item in cohorts
    )
    normalized = tuple(
        canonical_finalist_cohort_v1(
            item.primary_decision_score_evidence,
            cohort_index=index,
            evidence_hash=item.sha256,
            actual_action=item.actual_action,
            best_action=item.best_action,
        )
        for index, item in enumerate(cohorts)
    )
    pair_projection = project_finalist_adjudication_v2(
        normalized[:2],
        primary_best_action=primary.best_action,
        primary_actual_action=primary.actual_action,
        adjudication_profile=profile,
    )
    projection = project_finalist_adjudication_v2(
        normalized,
        primary_best_action=primary.best_action,
        primary_actual_action=primary.actual_action,
        adjudication_profile=profile,
        initial_reasons=pair_projection["strictReasons"],
    )
    primary_dict = primary.to_dict()
    reanalysis = cohorts[1]
    reanalysis_dict = reanalysis.to_dict()
    best_hashes = [
        canonical_sha256_v1(dict(item.best_action)) for item in cohorts
    ]
    return {
        "decisionId": primary.decision_id,
        "ordinal": primary.ordinal,
        **projection,
        "cohortCount": len(cohorts),
        "cohortAnalysisHashes": [item.sha256 for item in cohorts],
        "cohortAnalyses": [copy.deepcopy(item.to_dict()) for item in cohorts],
        "cohortBestActionHashes": best_hashes,
        "cohortHiddenDeals": [item.hidden_deals for item in cohorts],
        "cohortConfidence": [item["confidence"] for item in normalized],
        "cohortExhaustiveExactSupport": [
            bool(item["exhaustiveExactSupport"]) for item in classifications
        ],
        "cohortPopulationCoverageComplete": [
            bool(item["populationCoverageComplete"]) for item in classifications
        ],
        "cohortSolverCompletion": [
            str(item["solverCompletion"]) for item in classifications
        ],
        "cohortTruthClass": [
            str(item["truthClass"]) for item in classifications
        ],
        "primaryAnalysisHash": primary.sha256,
        "reanalysisHash": reanalysis.sha256,
        "primaryAnalysis": copy.deepcopy(primary_dict),
        "reanalysis": copy.deepcopy(reanalysis_dict),
        "primaryBestActionHash": best_hashes[0],
        "reanalysisBestActionHash": best_hashes[1],
        "selectedAnalysisHash": primary.sha256,
        "retainedPrimaryAnalysisHash": primary.sha256,
        "primaryAnalysisRetained": True,
        "primaryHiddenDeals": primary.hidden_deals,
        "reanalysisHiddenDeals": reanalysis.hidden_deals,
        "primaryConfidence": normalized[0]["confidence"],
        "reanalysisConfidence": normalized[1]["confidence"],
        "primaryExhaustiveExactSupport": bool(
            classifications[0]["exhaustiveExactSupport"]
        ),
        "reanalysisExhaustiveExactSupport": bool(
            classifications[1]["exhaustiveExactSupport"]
        ),
    }


def _closed_overflow_evaluation(
    evaluation: dict[str, object],
) -> dict[str, object]:
    return {
        "decisionId": evaluation["decisionId"],
        "ordinal": evaluation["ordinal"],
        "status": "unresolved",
        "strictRecommendationStatus": None,
        "strictPrimaryActionConfirmed": False,
        "reasons": ["adjudication_bound_exceeded"],
        "strictReasons": ["adjudication_bound_exceeded"],
        "competenceEvidence": None,
        "initialReasons": [],
        "cohortCount": 1,
        "agreementCount": 0,
        "supporterCohortIndices": [],
        "majorityActionHash": None,
        "cohortAnalysisHashes": [evaluation["cohortAnalysisHashes"][0]],
        "cohortAnalyses": [copy.deepcopy(evaluation["cohortAnalyses"][0])],
        "cohortBestActionHashes": [evaluation["cohortBestActionHashes"][0]],
        "cohortHiddenDeals": [evaluation["cohortHiddenDeals"][0]],
        "cohortConfidence": [evaluation["cohortConfidence"][0]],
        "cohortExhaustiveExactSupport": [
            evaluation["cohortExhaustiveExactSupport"][0]
        ],
        "cohortPopulationCoverageComplete": [
            evaluation["cohortPopulationCoverageComplete"][0]
        ],
        "cohortSolverCompletion": [evaluation["cohortSolverCompletion"][0]],
        "cohortTruthClass": [evaluation["cohortTruthClass"][0]],
        "primaryAnalysisHash": evaluation["primaryAnalysisHash"],
        "reanalysisHash": None,
        "primaryAnalysis": copy.deepcopy(evaluation["primaryAnalysis"]),
        "reanalysis": None,
        "primaryBestActionHash": evaluation["primaryBestActionHash"],
        "reanalysisBestActionHash": None,
        "selectedAnalysisHash": evaluation["selectedAnalysisHash"],
        "retainedPrimaryAnalysisHash": evaluation["retainedPrimaryAnalysisHash"],
        "primaryAnalysisRetained": True,
        "featuredTruthEligible": False,
        "surroundingVerified": False,
        "primaryHiddenDeals": evaluation["primaryHiddenDeals"],
        "reanalysisHiddenDeals": None,
        "primaryConfidence": evaluation["primaryConfidence"],
        "reanalysisConfidence": None,
        "primaryExhaustiveExactSupport": evaluation[
            "primaryExhaustiveExactSupport"
        ],
        "reanalysisExhaustiveExactSupport": None,
        "expectedMarginDrift": None,
    }


def _rebind_schema3_receipt(receipt: dict[str, object]) -> None:
    assert receipt["schemaVersion"] == 3
    adjudication = receipt["adjudicationEvidence"]
    assert isinstance(adjudication, dict)
    adjudication["sha256"] = canonical_sha256_v1({
        key: value for key, value in adjudication.items() if key != "sha256"
    })
    _rebind_trace_receipt(receipt)


def _rebind_trace_receipt(receipt: dict[str, object]) -> None:
    schema_version = int(receipt["schemaVersion"])
    decisions = receipt["decisions"]
    assert isinstance(decisions, list)
    laffa = receipt["laffaEvidence"]
    assert isinstance(laffa, dict)
    core = {
        "schemaVersion": schema_version,
        "traceHash": receipt["traceHash"],
        "profileHash": receipt["profileHash"],
        "analyzerId": receipt["analyzerId"],
        "analyzerVersion": receipt["analyzerVersion"],
        "decisionAnalysisHashes": [item["analysisHash"] for item in decisions],
        "laffaEvidenceHash": laffa["sha256"],
    }
    if schema_version >= 2:
        core.update({
            "confirmedDecisionIds": receipt["confirmedDecisionIds"],
            "screeningProfileHash": receipt["screeningProfileHash"],
            "confirmationProfileHash": receipt["confirmationProfileHash"],
        })
    if schema_version == 3:
        adjudication = receipt["adjudicationEvidence"]
        assert isinstance(adjudication, dict)
        core.update({
            "adjudicationEvidenceHash": adjudication["sha256"],
            "unresolvedDecisionIds": receipt["unresolvedDecisionIds"],
        })
    receipt["sha256"] = canonical_sha256_v1(core)


def _parse_current_adaptive_receipt(
    receipt: dict[str, object],
    *,
    ledger: DecisionLedgerV1,
    expected_profile_hash: str | None = None,
) -> TraceAnalysisResultV1:
    return TraceAnalysisResultV1.from_dict(
        receipt,
        ledger=ledger,
        expected_adaptive_profile_hash=(
            str(receipt["profileHash"])
            if expected_profile_hash is None
            else expected_profile_hash
        ),
    )


def _rebind_schema3_profiles(receipt: dict[str, object]) -> None:
    adjudication = receipt["adjudicationEvidence"]
    assert isinstance(adjudication, dict)
    profile = adjudication["adjudicationProfile"]
    assert isinstance(profile, dict)
    profile["sha256"] = canonical_sha256_v1({
        key: value for key, value in profile.items() if key != "sha256"
    })
    adjudication["profileHash"] = profile["sha256"]
    adjudication["primaryProfileHash"] = profile["primaryProfileHash"]
    receipt["profileHash"] = canonical_sha256_v1({
        "schemaVersion": 2,
        "primaryProfileHash": adjudication["primaryProfileHash"],
        "adjudicationProfileHash": adjudication["profileHash"],
        "selectionProfileHash": adjudication["selectionProfile"]["sha256"],
    })
    _rebind_schema3_receipt(receipt)


def _rebind_serialized_decision_analysis(analysis: dict[str, object]) -> None:
    score_evidence = analysis.get("primaryDecisionScoreEvidence")
    if isinstance(score_evidence, dict):
        score_evidence["rowsHash"] = decision_score_rows_hash_v1(
            str(score_evidence["decisionKind"]),
            score_evidence["rows"],
        )
        score_evidence["sha256"] = canonical_sha256_v1({
            key: value
            for key, value in score_evidence.items()
            if key != "sha256"
        })
        score_hash = score_evidence["sha256"]
    else:
        score_hash = None
    forced_proof = analysis.get("forcedLegalityProof")
    proof_hash = (
        forced_proof["sha256"] if isinstance(forced_proof, dict) else None
    )
    core = {
        "schemaVersion": 3,
        "decisionId": analysis["decisionId"],
        "ordinal": analysis["ordinal"],
        "actingSeat": analysis["actor"],
        "decisionKind": analysis["decisionKind"],
        "forced": analysis["forced"],
        "actualAction": analysis["actualAction"],
        "bestAction": analysis["bestAction"],
        "alternatives": analysis["alternatives"],
        "method": analysis["engineMethod"],
        "hiddenDeals": analysis["hiddenDeals"],
        "effectiveSampleSize": analysis["effectiveSampleSize"],
        "difficulty": analysis["predictedDifficulty"],
        "debateScore": analysis["debateScore"],
        "expectedRegret": analysis["expectedRegret"],
        "probabilityDrop": analysis["winProbabilityDrop"],
        "catastrophicCrossing": analysis["catastrophicCrossing"],
        "supportedMetrics": analysis["supportedMetrics"],
        "unsupportedMetrics": analysis["unsupportedMetrics"],
        "primaryDecisionScoreEvidenceHash": score_hash,
        "forcedLegalityProofHash": proof_hash,
    }
    analysis["analysisHash"] = canonical_sha256_v1(core)


def _replace_receipt_reanalysis(
    receipt: dict[str, object],
    reanalysis: dict[str, object],
) -> None:
    adjudication = receipt["adjudicationEvidence"]
    assert isinstance(adjudication, dict)
    evaluation_index = next(
        index
        for index, item in enumerate(adjudication["evaluations"])
        if item["decisionId"] == reanalysis["decisionId"]
    )
    evaluation = adjudication["evaluations"][evaluation_index]
    evaluation["cohortAnalyses"][1] = copy.deepcopy(reanalysis)
    evaluation["cohortAnalysisHashes"][1] = reanalysis["analysisHash"]
    evaluation["reanalysis"] = copy.deepcopy(reanalysis)
    evaluation["reanalysisHash"] = reanalysis["analysisHash"]
    _rebind_schema3_receipt(receipt)


def _robust_competence_receipt(
    receipt: dict[str, object],
    *,
    ledger: DecisionLedgerV1,
) -> str:
    adjudication = receipt["adjudicationEvidence"]
    assert isinstance(adjudication, dict)
    evaluation_index = next(
        index
        for index, item in enumerate(adjudication["evaluations"])
        if canonical_sha256_v1(item["cohortAnalyses"][0]["actualAction"])
        == canonical_sha256_v1(item["cohortAnalyses"][0]["bestAction"])
    )
    evaluation = adjudication["evaluations"][evaluation_index]
    primary = TraceDecisionAnalysisV3.from_dict(evaluation["cohortAnalyses"][0])
    reanalysis = copy.deepcopy(evaluation["cohortAnalyses"][1])
    score_evidence = reanalysis["primaryDecisionScoreEvidence"]
    best_row = score_evidence["rows"][0]
    drift = 2_000_000.0
    best_row["expectedRoundMargin"] = (
        float(best_row["expectedRoundMargin"]) + drift
    )
    best_row["downsideCvar10"] = float(best_row["downsideCvar10"]) + drift
    if "expectedTeamAwardedPoints" in best_row:
        best_row["expectedTeamAwardedPoints"] = (
            float(best_row["expectedTeamAwardedPoints"]) + drift
        )
    for alternative, row in zip(
        reanalysis["alternatives"], score_evidence["rows"], strict=True
    ):
        alternative.update({
            "rank": int(row["rank"]),
            "expectedMargin": float(row["expectedRoundMargin"]),
            "winProbability": float(row["roundLeadProbability"]),
            "bestShare": float(row["bestShare"]),
        })
    _rebind_serialized_decision_analysis(reanalysis)
    parsed_reanalysis = TraceDecisionAnalysisV3.from_dict(reanalysis)
    ledger_record = next(
        item for item in ledger.records if item.decision_id == primary.decision_id
    )
    third = copy.deepcopy(reanalysis)
    third_score = third["primaryDecisionScoreEvidence"]
    third_score["cohortSeed"] = _decision_seed(
        ledger_record,
        third_score["analysisIdentityHash"],
        seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
        seed_offset=209_458,
    )
    _rebind_serialized_decision_analysis(third)
    parsed_third = TraceDecisionAnalysisV3.from_dict(third)
    profile = adjudication["adjudicationProfile"]
    replacement = _canonical_adjudication_evaluation(
        primary,
        (primary, parsed_reanalysis, parsed_third),
        profile,
    )
    assert replacement["status"] == "primary_robust_competent"
    adjudication["evaluations"][evaluation_index] = replacement
    third_ids = set(adjudication["thirdCohortDecisionIds"])
    third_ids.add(primary.decision_id)
    adjudication["thirdCohortDecisionIds"] = [
        decision_id
        for decision_id in adjudication["targetDecisionIds"]
        if decision_id in third_ids
    ]
    adjudication["competentDecisionIds"] = sorted(
        item["decisionId"]
        for item in adjudication["evaluations"]
        if item["status"] == "primary_robust_competent"
    )
    adjudication["unresolvedDecisionIds"] = sorted(
        item["decisionId"]
        for item in adjudication["evaluations"]
        if item["status"] == "unresolved"
    )
    receipt["unresolvedDecisionIds"] = adjudication[
        "unresolvedDecisionIds"
    ]
    _rebind_schema3_receipt(receipt)
    return primary.decision_id


def test_trace_decision_parser_recomputes_unhashed_derived_projections(trace_bundle):
    trace, bundle = trace_bundle
    item = _analyzer(bundle, workers=1).analyze_decisions(
        trace,
        (next(record.decision_id for record in trace.decision_ledger.records if not record.forced),),
    )[0]

    mutated = item.to_dict()
    mutated["auditConfidence"] += 0.01
    with pytest.raises(ValueError, match="audit confidence mismatch"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["qualityScore"] += 0.01
    with pytest.raises(ValueError, match="quality score mismatch"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["reasons"] = [{"code": "invented_reason"}]
    with pytest.raises(ValueError, match="reason projection mismatch"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["unsupportedMetrics"].append(mutated["supportedMetrics"][0])
    mutated["unsupportedMetrics"].sort()
    with pytest.raises(ValueError, match="must be disjoint"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["alternatives"][0]["unknownDisplayField"] = 1
    _rebind_serialized_decision_analysis(mutated)
    with pytest.raises(ValueError, match="alternative .*unknown keys"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["ordinal"] = str(mutated["ordinal"])
    _rebind_serialized_decision_analysis(mutated)
    with pytest.raises(ValueError, match="ordinal must be a non-negative integer"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    for field in ("expectedRegret", "winProbabilityDrop"):
        mutated = item.to_dict()
        mutated[field] = float(mutated[field]) + 0.01
        _rebind_serialized_decision_analysis(mutated)
        with pytest.raises(ValueError, match="qualification metrics drift"):
            TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["catastrophicCrossing"] = not mutated["catastrophicCrossing"]
    _rebind_serialized_decision_analysis(mutated)
    with pytest.raises(ValueError, match="qualification metrics drift"):
        TraceDecisionAnalysisV3.from_dict(mutated)

    mutated = item.to_dict()
    mutated["alternatives"][0]["expectedMargin"] = (
        float(mutated["alternatives"][0]["expectedMargin"]) + 1.0
    )
    _rebind_serialized_decision_analysis(mutated)
    with pytest.raises(ValueError, match="alternatives do not mirror"):
        TraceDecisionAnalysisV3.from_dict(mutated)


def test_schema3_adjudication_receipt_is_closed_and_primary_retaining(trace_bundle):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )

    parsed = _parse_current_adaptive_receipt(
        receipt,
        ledger=trace.decision_ledger,
    )
    assert parsed.to_dict() == receipt
    with pytest.raises(ValueError, match="requires an expected effective profile"):
        TraceAnalysisResultV1.from_dict(
            receipt,
            ledger=trace.decision_ledger,
        )
    with pytest.raises(ValueError, match="does not match expected authority"):
        TraceAnalysisResultV1.from_dict(
            receipt,
            ledger=trace.decision_ledger,
            expected_adaptive_profile_hash="f" * 64,
        )

    mutated = copy.deepcopy(receipt)
    mutated["adjudicationEvidence"]["evaluations"][0]["status"] = "invented"
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="unsupported finalist adjudication status"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    evaluation = mutated["adjudicationEvidence"]["evaluations"][0]
    mutated["adjudicationEvidence"]["evaluations"].append(copy.deepcopy(evaluation))
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="not unique"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    decision_id = mutated["adjudicationEvidence"]["evaluations"][0]["decisionId"]
    mutated["adjudicationEvidence"]["unresolvedDecisionIds"] = [decision_id]
    mutated["adjudicationEvidence"]["competentDecisionIds"] = [decision_id]
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="unresolved precedence"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    evaluation = mutated["adjudicationEvidence"]["evaluations"][0]
    replacement = next(
        item for item in mutated["decisions"]
        if item["decisionId"] != evaluation["decisionId"]
    )
    evaluation["primaryAnalysis"] = copy.deepcopy(replacement)
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="primary analysis binding mismatch"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_schema3_competence_hash_and_status_are_bound(trace_bundle):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    competent_id = _robust_competence_receipt(
        receipt,
        ledger=trace.decision_ledger,
    )
    assert _parse_current_adaptive_receipt(
        receipt,
        ledger=trace.decision_ledger,
    ).to_dict() == receipt

    mutated = copy.deepcopy(receipt)
    competence = next(
        item
        for item in mutated["adjudicationEvidence"]["evaluations"]
        if item["decisionId"] == competent_id
    )["competenceEvidence"]
    competence["status"] = "unresolved"
    competence["sha256"] = canonical_sha256_v1({
        key: value for key, value in competence.items() if key != "sha256"
    })
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="competence status mismatch"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_trace_result_envelope_order_and_confirmation_set_are_closed(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )

    unknown = copy.deepcopy(receipt)
    unknown["unknownResultField"] = True
    _rebind_schema3_receipt(unknown)
    with pytest.raises(ValueError, match="unknown keys"):
        _parse_current_adaptive_receipt(unknown, ledger=trace.decision_ledger)

    schema_alias = copy.deepcopy(receipt)
    schema_alias["schemaVersion"] = "3"
    _rebind_trace_receipt(schema_alias)
    with pytest.raises(ValueError, match="unsupported trace analysis result schema"):
        _parse_current_adaptive_receipt(
            schema_alias,
            ledger=trace.decision_ledger,
            expected_profile_hash=str(receipt["profileHash"]),
        )

    permuted = copy.deepcopy(receipt)
    permuted["decisions"].reverse()
    _rebind_schema3_receipt(permuted)
    with pytest.raises(ValueError, match="exact canonical decision ledger"):
        _parse_current_adaptive_receipt(permuted, ledger=trace.decision_ledger)

    nonforced_ids = sorted(
        item["decisionId"] for item in receipt["decisions"] if not item["forced"]
    )
    duplicated = copy.deepcopy(receipt)
    duplicated["confirmedDecisionIds"] = [nonforced_ids[0], nonforced_ids[0]]
    _rebind_schema3_receipt(duplicated)
    with pytest.raises(ValueError, match="canonical decision-id set"):
        _parse_current_adaptive_receipt(duplicated, ledger=trace.decision_ledger)

    forced_id = next(
        item["decisionId"] for item in receipt["decisions"] if item["forced"]
    )
    forced = copy.deepcopy(receipt)
    forced["confirmedDecisionIds"] = [forced_id]
    _rebind_schema3_receipt(forced)
    with pytest.raises(ValueError, match="must be non-forced"):
        _parse_current_adaptive_receipt(forced, ledger=trace.decision_ledger)

    bounded = copy.deepcopy(receipt)
    bounded["confirmedDecisionIds"] = nonforced_ids[:2]
    primary_profile = bounded["adjudicationEvidence"]["primaryProfile"]
    primary_profile["maximumConfirmationDecisionsPerTrace"] = 1
    primary_profile["sha256"] = canonical_sha256_v1({
        key: value for key, value in primary_profile.items() if key != "sha256"
    })
    bounded["adjudicationEvidence"]["adjudicationProfile"][
        "primaryProfileHash"
    ] = primary_profile["sha256"]
    _rebind_schema3_profiles(bounded)
    with pytest.raises(ValueError, match="reconstructed primary profile bound"):
        _parse_current_adaptive_receipt(bounded, ledger=trace.decision_ledger)


def test_same_profile_stage_identities_roundtrip_and_drift_per_stage(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    records = {item.decision_id: item for item in trace.decision_ledger.records}

    def replace_primary_identity(
        target: dict[str, object],
        decision_id: str,
        identity_hash: str,
    ) -> None:
        decision_index = next(
            index
            for index, item in enumerate(target["decisions"])
            if item["decisionId"] == decision_id
        )
        primary_value = copy.deepcopy(target["decisions"][decision_index])
        score = primary_value["primaryDecisionScoreEvidence"]
        score["analysisIdentityHash"] = identity_hash
        score["cohortSeed"] = _decision_seed(
            records[decision_id],
            identity_hash,
            seed_namespace="",
            seed_offset=0,
        )
        _rebind_serialized_decision_analysis(primary_value)
        parsed_primary = TraceDecisionAnalysisV3.from_dict(primary_value)
        target["decisions"][decision_index] = primary_value

        adjudication = target["adjudicationEvidence"]
        evaluation_index = next(
            index
            for index, item in enumerate(adjudication["evaluations"])
            if item["decisionId"] == decision_id
        )
        evaluation = adjudication["evaluations"][evaluation_index]
        cohort_values = copy.deepcopy(evaluation["cohortAnalyses"])
        cohort_values[0] = primary_value
        cohorts = tuple(
            TraceDecisionAnalysisV3.from_dict(item) for item in cohort_values
        )
        adjudication["evaluations"][evaluation_index] = (
            _canonical_adjudication_evaluation(
                parsed_primary,
                cohorts,
                adjudication["adjudicationProfile"],
            )
        )
        adjudication["competentDecisionIds"] = sorted(
            item["decisionId"]
            for item in adjudication["evaluations"]
            if item["status"] == "primary_robust_competent"
        )
        adjudication["unresolvedDecisionIds"] = sorted(
            item["decisionId"]
            for item in adjudication["evaluations"]
            if item["status"] == "unresolved"
        )
        target["unresolvedDecisionIds"] = adjudication[
            "unresolvedDecisionIds"
        ]

    evaluation_ids = [
        item["decisionId"]
        for item in receipt["adjudicationEvidence"]["evaluations"]
    ]
    confirmation_id, screening_id = evaluation_ids[:2]
    replace_primary_identity(receipt, confirmation_id, "c" * 64)
    receipt["confirmedDecisionIds"] = [confirmation_id]
    _rebind_schema3_receipt(receipt)

    assert _parse_current_adaptive_receipt(
        receipt,
        ledger=trace.decision_ledger,
    ).to_dict() == receipt

    drifted = copy.deepcopy(receipt)
    replace_primary_identity(drifted, screening_id, "d" * 64)
    _rebind_schema3_receipt(drifted)
    with pytest.raises(ValueError, match="primary stage identity drift"):
        _parse_current_adaptive_receipt(drifted, ledger=trace.decision_ledger)


def test_schema3_reanalysis_authority_substitution_fails_after_full_rehash(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    evaluation = receipt["adjudicationEvidence"]["evaluations"][0]
    ledger_record = next(
        item
        for item in trace.decision_ledger.records
        if item.decision_id == evaluation["decisionId"]
    )
    primary_identity = evaluation["cohortAnalyses"][0][
        "primaryDecisionScoreEvidence"
    ]["analysisIdentityHash"]

    mutated = copy.deepcopy(receipt)
    reanalysis = mutated["adjudicationEvidence"]["evaluations"][0][
        "cohortAnalyses"
    ][1]
    reanalysis["primaryDecisionScoreEvidence"][
        "traceAnalysisProfileHash"
    ] = "f" * 64
    _rebind_serialized_decision_analysis(reanalysis)
    _replace_receipt_reanalysis(mutated, reanalysis)
    with pytest.raises(ValueError, match="reanalysis profile mismatch"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    reanalysis = mutated["adjudicationEvidence"]["evaluations"][0][
        "cohortAnalyses"
    ][1]
    reanalysis["primaryDecisionScoreEvidence"][
        "analysisIdentityHash"
    ] = primary_identity
    reanalysis["primaryDecisionScoreEvidence"]["cohortSeed"] = _decision_seed(
        ledger_record,
        primary_identity,
        seed_namespace="finalist_surrounding_adjudication_v2",
        seed_offset=104_729,
    )
    _rebind_serialized_decision_analysis(reanalysis)
    _replace_receipt_reanalysis(mutated, reanalysis)
    with pytest.raises(ValueError, match="lane identities must be distinct"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    reanalysis = mutated["adjudicationEvidence"]["evaluations"][0][
        "cohortAnalyses"
    ][1]
    reanalysis["primaryDecisionScoreEvidence"]["cohortIndex"] = 0
    _rebind_serialized_decision_analysis(reanalysis)
    _replace_receipt_reanalysis(mutated, reanalysis)
    with pytest.raises(ValueError, match="reanalysis cohort role mismatch"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_schema3_third_cohort_reanalysis_identity_drift_fails_closed(trace_bundle):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    evaluation = receipt["adjudicationEvidence"]["evaluations"][0]
    ledger_record = next(
        item
        for item in trace.decision_ledger.records
        if item.decision_id == evaluation["decisionId"]
    )
    third = copy.deepcopy(evaluation["cohortAnalyses"][1])
    third["primaryDecisionScoreEvidence"]["analysisIdentityHash"] = "1" * 64
    third["primaryDecisionScoreEvidence"]["cohortSeed"] = _decision_seed(
        ledger_record,
        "1" * 64,
        seed_namespace="finalist_surrounding_adjudication_v2:cohort3",
        seed_offset=209_458,
    )
    _rebind_serialized_decision_analysis(third)
    classification = canonical_score_support_classification_v1(
        third["primaryDecisionScoreEvidence"]
    )
    evaluation["cohortCount"] = 3
    evaluation["cohortAnalyses"].append(third)
    evaluation["cohortAnalysisHashes"].append(third["analysisHash"])
    evaluation["cohortBestActionHashes"].append(
        canonical_sha256_v1(third["bestAction"])
    )
    evaluation["cohortHiddenDeals"].append(third["hiddenDeals"])
    evaluation["cohortConfidence"].append(third["auditConfidence"])
    evaluation["cohortExhaustiveExactSupport"].append(
        bool(classification["exhaustiveExactSupport"])
    )
    evaluation["cohortPopulationCoverageComplete"].append(
        bool(classification["populationCoverageComplete"])
    )
    evaluation["cohortSolverCompletion"].append(
        str(classification["solverCompletion"])
    )
    evaluation["cohortTruthClass"].append(str(classification["truthClass"]))
    receipt["adjudicationEvidence"]["thirdCohortDecisionIds"] = [
        evaluation["decisionId"]
    ]
    _rebind_schema3_receipt(receipt)

    with pytest.raises(ValueError, match="reanalysis identity drift"):
        _parse_current_adaptive_receipt(receipt, ledger=trace.decision_ledger)


def test_schema3_profiles_and_exact_target_coverage_survive_full_rehash(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )

    mutated = copy.deepcopy(receipt)
    primary_profile = mutated["adjudicationEvidence"]["primaryProfile"]
    primary_profile["screeningProfileHash"] = "f" * 64
    primary_profile["sha256"] = canonical_sha256_v1({
        key: value for key, value in primary_profile.items() if key != "sha256"
    })
    adjudication_profile = mutated["adjudicationEvidence"][
        "adjudicationProfile"
    ]
    adjudication_profile["primaryProfileHash"] = primary_profile["sha256"]
    _rebind_schema3_profiles(mutated)
    with pytest.raises(ValueError, match="primary profile snapshot binding mismatch"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    adjudication_profile = mutated["adjudicationEvidence"][
        "adjudicationProfile"
    ]
    adjudication_profile["maximumExpectedMarginDrift"] = int(
        adjudication_profile["maximumExpectedMarginDrift"]
    )
    _rebind_schema3_profiles(mutated)
    with pytest.raises(ValueError, match="normalized floating-point"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    selection = mutated["adjudicationEvidence"]["selectionProfile"]
    selection["minimumSurroundingConfidence"] = 1.0
    selection_value = {
        key: value for key, value in selection.items() if key != "sha256"
    }
    mutated["adjudicationEvidence"]["selectionProfile"] = (
        SelectionProfileV1.build(selection_value).to_dict()
    )
    mutated["adjudicationEvidence"]["adjudicationProfile"][
        "minimumConfidence"
    ] = 1.0
    _rebind_schema3_profiles(mutated)
    with pytest.raises(
        ValueError,
        match="verdict does not match canonical score rows|third-cohort coverage",
    ):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    relaxed = copy.deepcopy(receipt)
    selection = relaxed["adjudicationEvidence"]["selectionProfile"]
    selection["maximumSurroundingMaterialMistakes"] = 1
    selection_value = {
        key: value for key, value in selection.items() if key != "sha256"
    }
    relaxed["adjudicationEvidence"]["selectionProfile"] = (
        SelectionProfileV1.build(selection_value).to_dict()
    )
    _rebind_schema3_profiles(relaxed)
    assert _parse_current_adaptive_receipt(
        relaxed,
        ledger=trace.decision_ledger,
    ).to_dict() == relaxed
    with pytest.raises(ValueError, match="does not match expected authority"):
        _parse_current_adaptive_receipt(
            relaxed,
            ledger=trace.decision_ledger,
            expected_profile_hash=str(receipt["profileHash"]),
        )

    mutated = copy.deepcopy(receipt)
    adjudication = mutated["adjudicationEvidence"]
    for field in (
        "targetDecisionIds",
        "boundedDecisionIds",
        "overflowDecisionIds",
        "thirdCohortDecisionIds",
        "evaluations",
        "unresolvedDecisionIds",
        "competentDecisionIds",
    ):
        adjudication[field] = []
    mutated["unresolvedDecisionIds"] = []
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="target coverage is incomplete"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_schema3_recomputes_strict_and_competence_numeric_projections(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    competent_id = _robust_competence_receipt(
        receipt,
        ledger=trace.decision_ledger,
    )

    mutated = copy.deepcopy(receipt)
    evaluation = next(
        item
        for item in mutated["adjudicationEvidence"]["evaluations"]
        if item["decisionId"] == competent_id
    )
    evaluation["expectedMarginDrift"] += 1.0
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="verdict does not match canonical score rows"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    evaluation = next(
        item
        for item in mutated["adjudicationEvidence"]["evaluations"]
        if item["decisionId"] == competent_id
    )
    evaluation["supporterCohortIndices"] = list(
        reversed(evaluation["supporterCohortIndices"])
    )
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="verdict does not match canonical score rows"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    evaluation = next(
        item
        for item in mutated["adjudicationEvidence"]["evaluations"]
        if item["decisionId"] == competent_id
    )
    competence = evaluation["competenceEvidence"]
    competence["maximumExpectedMarginDrift"] += 1.0
    competence["sha256"] = canonical_sha256_v1({
        key: value for key, value in competence.items() if key != "sha256"
    })
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="verdict does not match canonical score rows"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    evaluation = next(
        item
        for item in mutated["adjudicationEvidence"]["evaluations"]
        if item["decisionId"] == competent_id
    )
    competence = evaluation["competenceEvidence"]
    competence["cohortEvaluations"][0]["expectedMarginRegret"] += 0.25
    competence["expectedMarginRegretDrift"] = 0.25
    competence["sha256"] = canonical_sha256_v1({
        key: value for key, value in competence.items() if key != "sha256"
    })
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="verdict does not match canonical score rows"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_schema3_overflow_roundtrip_is_closed_and_normalized(trace_bundle):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )
    adjudication = receipt["adjudicationEvidence"]
    target_ids = list(adjudication["targetDecisionIds"])
    assert len(target_ids) > 1
    adjudication["boundedDecisionIds"] = target_ids[:1]
    adjudication["overflowDecisionIds"] = target_ids[1:]
    adjudication["thirdCohortDecisionIds"] = [
        decision_id
        for decision_id in adjudication["thirdCohortDecisionIds"]
        if decision_id == target_ids[0]
    ]
    adjudication["evaluations"] = [
        evaluation
        if evaluation["decisionId"] == target_ids[0]
        else _closed_overflow_evaluation(evaluation)
        for evaluation in adjudication["evaluations"]
    ]
    adjudication["competentDecisionIds"] = sorted(
        evaluation["decisionId"]
        for evaluation in adjudication["evaluations"]
        if evaluation["status"] == "primary_robust_competent"
    )
    adjudication["unresolvedDecisionIds"] = sorted(
        evaluation["decisionId"]
        for evaluation in adjudication["evaluations"]
        if evaluation["status"] == "unresolved"
    )
    receipt["unresolvedDecisionIds"] = adjudication["unresolvedDecisionIds"]
    adjudication["adjudicationProfile"]["maximumDecisionsPerTrace"] = 1
    _rebind_schema3_profiles(receipt)

    parsed = _parse_current_adaptive_receipt(
        receipt,
        ledger=trace.decision_ledger,
    )
    assert parsed.to_dict() == receipt
    assert all(
        evaluation["cohortCount"] == 1
        and evaluation["reasons"] == ["adjudication_bound_exceeded"]
        for evaluation in adjudication["evaluations"][1:]
    )

    mutated = copy.deepcopy(receipt)
    overflow = mutated["adjudicationEvidence"]["evaluations"][1]
    overflow["agreementCount"] = 1
    overflow["supporterCohortIndices"] = [0]
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="overflow evidence is invalid"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)

    mutated = copy.deepcopy(receipt)
    mutated["adjudicationEvidence"]["evaluations"][1][
        "majorityActionHash"
    ] = "f" * 64
    _rebind_schema3_receipt(mutated)
    with pytest.raises(ValueError, match="overflow evidence is invalid"):
        _parse_current_adaptive_receipt(mutated, ledger=trace.decision_ledger)


def test_current_schema_downgrade_and_historical_qualification_fail_closed(
    trace_bundle,
):
    trace, bundle = trace_bundle
    result = _analyzer(bundle, workers=1).analyze_trace(trace)
    receipt = _strict_schema3_receipt(
        result,
        _receipt_reanalysis(trace, bundle, result),
        ledger=trace.decision_ledger,
    )

    schema2 = copy.deepcopy(receipt)
    schema2["schemaVersion"] = 2
    schema2.pop("adjudicationEvidence")
    schema2.pop("unresolvedDecisionIds")
    _rebind_trace_receipt(schema2)
    with pytest.raises(ValueError, match="requires adjudication-v2 evidence"):
        TraceAnalysisResultV1.from_dict(schema2, ledger=trace.decision_ledger)

    schema2["analyzerVersion"] = "1.2.0"
    _rebind_trace_receipt(schema2)
    historical = TraceAnalysisResultV1.from_dict(
        schema2,
        ledger=trace.decision_ledger,
    )
    expected_unresolved = tuple(sorted(
        item.decision_id for item in historical.decisions if not item.forced
    ))
    assert historical.adjudication_evidence is None
    assert historical.unresolved_decision_ids == expected_unresolved

    schema3_v1 = copy.deepcopy(receipt)
    schema3_v1["analyzerVersion"] = "1.2.0"
    schema3_v1["unresolvedDecisionIds"] = list(expected_unresolved)
    schema3_v1["adjudicationEvidence"] = {
        "schemaVersion": 1,
        "evaluations": [],
        "competentDecisionIds": [],
        "unresolvedDecisionIds": list(expected_unresolved),
    }
    _rebind_schema3_receipt(schema3_v1)
    parsed_v1 = TraceAnalysisResultV1.from_dict(
        schema3_v1,
        ledger=trace.decision_ledger,
    )
    assert parsed_v1.unresolved_decision_ids == expected_unresolved

    schema3_v1["adjudicationEvidence"]["evaluations"] = [{
        "status": "primary_robust_competent",
        "competenceEvidence": {
            "status": "primary_robust_competent",
            "cohortEvaluations": [{"qualified": True}],
        },
    }]
    _rebind_schema3_receipt(schema3_v1)
    with pytest.raises(ValueError, match="cannot carry current competent"):
        TraceAnalysisResultV1.from_dict(
            schema3_v1,
            ledger=trace.decision_ledger,
        )

    duplicated = copy.deepcopy(receipt)
    duplicated["decisions"].append(copy.deepcopy(duplicated["decisions"][0]))
    _rebind_schema3_receipt(duplicated)
    with pytest.raises(ValueError, match="exact .*decision ledger"):
        _parse_current_adaptive_receipt(
            duplicated,
            ledger=trace.decision_ledger,
        )

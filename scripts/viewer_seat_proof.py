"""Compute exact per-deal double-dummy R and D for a replayable play scenario.

The viewer's legal deals are exhausted before any score is emitted. Continuations
use full information within each deal, so this receipt is separate from the
observer-safe policy certificate checked by actor_interest_primary_failures_v1.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import ortools

from baloot_platform.analysis.double_dummy import DoubleDummySolver
from baloot_platform.analysis.hidden_deals import HiddenDealBudgetExceeded, generate_hidden_deals
from baloot_platform.analysis.models import InformationState
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.rules import RuleBook
from baloot_platform.native.accelerator import NativeUnavailable
from baloot_platform.research.exact_state import ExactStateNormalizerV1
from baloot_platform.research.ortools_proof import (
    OrToolsScenarioGraphProofV1, ProofGraphEdgeV1, ProofGraphNodeV1, ProofGraphV1,
)
from baloot_platform.research.scenario_contracts import ScenarioProofRequestV1, canonical_sha256


RULES_PATH = Path(__file__).resolve().parent.parent / "configs/rules/saudi_common_v1.yaml"


class WorkLimit(RuntimeError):
    pass


def cp_sat_values(engine, state, maximum_nodes, cp_work):
    """Encode the engine's complete legal game tree in the existing CP-SAT prover."""
    actor = state.current_player
    visited = 0
    nodes = []

    def visit():
        nonlocal visited
        if visited == maximum_nodes:
            raise WorkLimit("maximum_nodes")
        node_id = str(visited)
        visited += 1
        seat = state.current_player
        if seat is None:
            points = state.round_score.awarded_points
            value = points[int(actor.team)] - points[int(actor.team.other)]
            nodes.append(ProofGraphNodeV1.leaf(node_id, value))
            return node_id
        edges = []
        for action in engine.legal_play_actions(state):
            token = engine.apply(state, action.card, ekka=action.ekka, verify_identity=False, emit_events=False)
            try:
                child = visit()
            finally:
                engine.undo(state, token)
            edges.append(ProofGraphEdgeV1.build(action.action_id, action.code, child))
        nodes.append(ProofGraphNodeV1.decision(node_id, seat.team.name.lower(), edges))
        return node_id

    graph = ProofGraphV1.build(scope="exact", root_node_id=visit(), nodes=nodes)
    request = ScenarioProofRequestV1.build({
        "schemaVersion": 1, "mode": "exact", "visibility": "complete_state",
        "claim": "winning_root_action_count", "targetTeam": actor.team.name.lower(),
        "utility": "awarded_point_margin", "successThreshold": 0, "budgetProfile": "viewer_seat_proof_v1",
    })
    proof = OrToolsScenarioGraphProofV1().prove(
        graph, request, native_action_quality=True, maximum_seconds=math.inf,
        maximum_deterministic_time=cp_work,
    )
    if proof.completion_status == "model_error":
        raise ValueError("CP-SAT rejected the legal continuation graph")
    if proof.solver_status != "optimal" or proof.completion_status not in {"exact_complete", "proof_failed"}:
        raise WorkLimit("maximum_cp_work_per_deal")
    return {row.action_code: row.value for row in proof.root_actions}, visited


def prove_fixture(raw, *, fixture_sha, source_sha, maximum_allocations, maximum_nodes, cp_work,
                  backend="auto", native_library=None):
    if hashlib.sha256(raw).hexdigest() != fixture_sha:
        raise ValueError("scenario SHA-256 mismatch")
    if not 1 <= maximum_allocations <= 25000 or not 1 <= maximum_nodes < 2**64 - 1 or not 0 < cp_work <= 100:
        raise ValueError("allocation, node or CP-SAT work allowance is out of range")
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("a full source commit SHA is required")
    fixture = json.loads(raw)
    if not isinstance(fixture, dict):
        raise ValueError("the scenario must be a JSON object")
    prefix = fixture.get("exactTracePrefix") or fixture.get("origin", {}).get("exactTracePrefix")
    if not isinstance(prefix, dict):
        raise ValueError("the scenario must carry an exact play trace prefix")
    selected = fixture.get("expected", {}).get("selectedAction") or fixture.get("proof", {}).get("selectedAction")
    rules = RuleBook.load(RULES_PATH)
    exact = ExactStateNormalizerV1(rules).normalize(prefix, decision_point={"kind": "play"})
    state = exact.round_state
    rules_meaning = fixture.get("rulesMeaning")
    if (state.current_player is None or not isinstance(rules_meaning, dict)
            or rules_meaning.get("actingSeat") != state.current_player.name.lower()):
        raise ValueError("scenario rulesMeaning.actingSeat does not match the replayed viewer seat")
    engine = BalootEngine(rules)
    legal_plays = engine.legal_play_actions(state)
    if len({action.card for action in legal_plays}) == 1:
        raise ValueError("D is undefined when only one card is legal")
    actions = sorted(action.code for action in legal_plays)
    if selected not in actions:
        raise ValueError("the scenario must name a legal selected action")
    info = InformationState.from_complete(state, state.current_player)

    solver = None
    if backend != "ortools":
        try:
            solver = DoubleDummySolver(rules, native_library)
        except NativeUnavailable:
            if backend == "native" or native_library is not None:
                raise
    receipt = {
        "schemaVersion": 2, "scope": "exhaustive_dealwise_double_dummy", "status": "INCOMPLETE",
        "rMeaning": "R is the share of rule-consistent hidden deals where the selected action has the highest value, including ties. "
                    "Each deal counts equally and bids are not used as constraints. R is not a probability.",
        "utility": "awarded_point_margin", "selectedAction": selected,
        "observerSafePolicyProof": "NOT_RUN",
        "sourceCommit": source_sha, "scenarioSha256": fixture_sha, "rulesSha256": rules.sha256,
        "viewerInformationSha256": canonical_sha256(info.to_dict()),
        "backend": "native_dd_mtdf_v1" if solver else "ortools_graph_minimax",
        **({"nativeLibrarySha256": solver.library_hash} if solver else {"ortoolsVersion": ortools.__version__}),
        "budget": {"maximumAllocations": maximum_allocations, "maximumNodes": maximum_nodes,
                   **({} if solver else {"maximumCpWorkPerDeal": cp_work})},
        "coverageComplete": False, "dealsSolved": 0, "R": None, "D": None, "moves": [],
    }
    try:
        try:
            deals = generate_hidden_deals(info, engine, requested_samples=1, maximum_attempts=1, seed=0,
                                          exact_allocation_limit=maximum_allocations, exact_only=True)
        except HiddenDealBudgetExceeded as error:
            return {**receipt, "reason": "maximum_allocations", "allocationCandidateBound": error.candidate_bound}
        if not deals.coverage_complete:
            raise ValueError("exact-only deal enumeration returned partial coverage")
        receipt.update(allocationMethod=deals.method, allocationsAttempted=deals.attempted,
                       dealsTotal=len(deals.states), allocationRejectionsByReason=deals.rejections_by_reason)
        sums = dict.fromkeys(actions, 0)
        best = dict.fromkeys(actions, 0)
        strict = dict.fromkeys(actions, 0)
        nodes = 0
        evidence = hashlib.sha256()
        for world in sorted(deals.states, key=lambda item: item.hands):
            remaining = maximum_nodes - nodes
            try:
                if not remaining:
                    raise WorkLimit("maximum_nodes")
                if solver:
                    outcome = solver.solve(world, maximum_seconds=None, maximum_nodes=remaining, all_root_values=True)
                    if outcome["status"] != "exact":
                        if outcome["stopReason"] != "maximum_nodes":
                            raise RuntimeError("node-only native solve stopped without exhausting its budget")
                        raise WorkLimit("maximum_nodes")
                    values = {row["action"]: row["value"] for row in outcome["rootActions"]}
                    used = outcome["nodes"]
                else:
                    values, used = cp_sat_values(engine, world, remaining, cp_work)
            except WorkLimit as error:
                return {**receipt, "nodesInSolvedDeals": nodes, "reason": str(error)}
            if set(values) != set(actions) or any(type(value) is not int for value in values.values()):
                raise ValueError("exact backend did not return every legal root value")
            maximum = max(values.values())
            sole_winner = sum(value == maximum for value in values.values()) == 1
            for action, value in values.items():
                sums[action] += value
                best[action] += value == maximum
                strict[action] += value == maximum and sole_winner
            evidence.update(receipt_bytes([world.hands, values]))
            receipt["dealsSolved"] += 1
            nodes += used
    finally:
        if solver:
            solver.close()

    count = receipt["dealsSolved"]
    moves = []
    for action in actions:
        alternative = max(value for key, value in sums.items() if key != action)
        gap = sums[action] - alternative
        moves.append({"action": action, "bestDeals": best[action], "bestShare": best[action] / count,
                      "strictBestDeals": strict[action], "strictShare": strict[action] / count,
                      "meanGapNumerator": gap, "meanGap": gap / count, "uniqueBest": gap > 0})
    chosen = next(move for move in moves if move["action"] == selected)
    return {**receipt, "status": "exact_complete", "coverageComplete": True, "nodesInSolvedDeals": nodes,
            "dealRootValuesSha256": evidence.hexdigest(), "R": chosen["bestShare"], "D": chosen["meanGap"],
            "uniqueBest": chosen["uniqueBest"], "moves": moves,
            "scorerInputs": {"worldsComplete": True, "bestShare": chosen["bestShare"],
                             "strictShare": chosen["strictShare"], "worldMeanGap": chosen["meanGap"]}}


def receipt_bytes(receipt):
    return (json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--backend", choices=("auto", "native", "ortools"), default="auto")
    parser.add_argument("--maximum-allocations", type=int, default=4096)
    parser.add_argument("--maximum-nodes", type=int, default=2_000_000)
    parser.add_argument("--cp-work-per-deal", type=float, default=1)
    args = parser.parse_args()
    repo = RULES_PATH.parents[2]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, timeout=10).strip()
    if subprocess.run(["git", "diff", "--quiet", source, "--", "configs", "native", "scripts", "src", "pyproject.toml"],
                      cwd=repo, timeout=10).returncode:
        parser.error("proof source must match the checked-out commit")
    receipt = prove_fixture(args.fixture.read_bytes(), fixture_sha=args.fixture_sha256, source_sha=source,
                            maximum_allocations=args.maximum_allocations, maximum_nodes=args.maximum_nodes,
                            cp_work=args.cp_work_per_deal, backend=args.backend, native_library=args.native_library)
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_bytes(receipt_bytes(receipt))
    return 0 if receipt["status"] == "exact_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())

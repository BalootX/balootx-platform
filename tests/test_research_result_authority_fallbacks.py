from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from baloot_platform.research.result_authority import (
    ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES,
    ResearchAuthorityV1,
    _terminal_proof_source_v3,
    build_ortools_research_result_v3,
    build_ortools_result_identity_v2,
    validate_ortools_result_artifact_documents_v3,
    validate_ortools_result_sections_v1,
    validate_ortools_stage_receipt_v2,
)
from baloot_platform.research.scenario_contracts import canonical_sha256


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signed(core: dict) -> dict:
    return {**core, "sha256": canonical_sha256(core)}


def _json_bytes(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _authority(backend_id: str) -> ResearchAuthorityV1:
    return ResearchAuthorityV1.build(
        backend_id=backend_id,
        backend_version="3.0.0" if backend_id.endswith("v3") else "2.0.0",
        solver_version="9.15.0",
        bundle_revision_id="CBR-AUTHORITY-FALLBACK",
        bundle_hash=_hash("bundle"),
        domain_ir_hashes={"rules": _hash("domain")},
        lowering_manifest_hash=_hash("lowering"),
        construction_model_hash=_hash("construction"),
        proof_model_hash=_hash("proof-model"),
        graph_hash=_hash("proof-graph"),
        trace_hash=_hash("trace"),
        proof_outcome_hash=_hash("proof-outcome"),
    )


def _campaign_document(executor_hash: str) -> dict:
    semantic_core = {
        "schemaVersion": 1,
        "status": "not_found_within_budget",
        "requestedResultCount": 1,
        "maximumCandidates": 1,
        "symmetryPolicyId": "rules_preserving_v1",
        "distancePolicyId": "decision_state_card_hamming_v1",
        "minimumDistance": 1,
        "executionSemanticHashes": [executor_hash],
        "acceptedSemanticHashes": [],
        "variantReceipts": [],
        "noGoodHashes": [],
    }
    semantic_hash = canonical_sha256(semantic_core)
    receipt_core = {
        **semantic_core,
        "semanticHash": semantic_hash,
        "wallTimeSeconds": 0.25,
        "cpuSeconds": 0.2,
    }
    return {
        "schemaVersion": 1,
        "status": "not_found_within_budget",
        "requestedResultCount": 1,
        "maximumCandidates": 1,
        "symmetryPolicyId": "rules_preserving_v1",
        "distancePolicyId": "decision_state_card_hamming_v1",
        "minimumDistance": 1,
        "completedCandidates": 1,
        "executionHashes": [_hash("execution-v3")],
        "executionSemanticHashes": [executor_hash],
        "acceptedExecutionHashes": [],
        "acceptedSemanticHashes": [],
        "noGoodVectors": [],
        "variantReceipts": [],
        "wallTimeSeconds": 0.25,
        "cpuSeconds": 0.2,
        "semanticHash": semantic_hash,
        "sha256": canonical_sha256(receipt_core),
    }


def _v3_fixture(
    *,
    compilation_receipt: bool = False,
    candidate_without_normalization: bool = False,
    divergent_replay_valid_candidate: bool = False,
    raw_proof: dict | None = None,
    raw_model: dict | None = None,
    terminal_status: str | None = None,
    predicate_expression: dict | None = None,
    predicate_evaluation: dict | None = None,
    graph_utility_scale: int | None = None,
    graph_scope: str | None = None,
    backend_version: str | None = None,
    tactical_profile: dict | None = None,
) -> tuple[dict, dict, dict, dict]:
    bundle_hash = _hash("bundle-v3")
    rules_hash = _hash("rules-v3")
    domain_hash = _hash("domain-v3")
    configuration = {
        "schemaVersion": 1,
        "bundleRevisionId": "CBR-V3-IMMUTABLE",
        "semanticBundleHash": bundle_hash,
        "identities": {
            "rules": {"sha256": rules_hash},
            "domain": {"sha256": domain_hash},
        },
    }
    scenario = _signed(
        {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": predicate_expression if predicate_expression is not None else {"op": "all", "children": []},
            "decisionPoint": {"kind": "play"},
            "outcome": {"op": "all", "children": []},
        }
    )
    proof = _signed(
        {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "strategyClass": "pure_deterministic",
            "claim": "winning_root_action_count",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": 1.0,
            "lineUniverse": "all_legal",
        }
    )
    recipe_core = {
        "schemaVersion": 4,
        "code": "RCP-V3-AUTHORITY",
        "configuration": configuration,
        "scenario": scenario,
        "proof": proof,
    }
    recipe = {**recipe_core, "sha256": canonical_sha256(recipe_core)}
    run = {
        "runCode": "RUN-V3-AUTHORITY",
        "runHash": _hash("run-v3"),
        "recipeCode": recipe["code"],
        "recipeHash": recipe["sha256"],
        "configurationAuthority": copy.deepcopy(configuration),
        "createdAt": "2026-08-30T00:00:00Z",
    }
    documents = {
        "proofModel": {"schemaVersion": 1, "status": "not_compiled"},
        "proofGraph": {"schemaVersion": 1, "status": "not_compiled"},
        "completeTrace": {"schemaVersion": 1, "status": "not_available"},
        "proofOutcome": {"schemaVersion": 1, "status": "not_run"},
    }
    if graph_utility_scale is not None:
        documents["proofGraph"]["utilityScale"] = graph_utility_scale
    if raw_model is not None:
        documents["proofModel"] = copy.deepcopy(raw_model)
    if graph_scope is not None:
        documents["proofGraph"]["scope"] = graph_scope
    if raw_proof is not None:
        documents["proofOutcome"] = {key: copy.deepcopy(value)
                                     for key, value in raw_proof.items() if key != "sha256"}
    execution_status = {
        "proof_failed": "proof_failed", "proof_incomplete": "incomplete",
    }.get(terminal_status)
    if divergent_replay_valid_candidate:
        documents["completeTrace"]["exactNormalization"] = {
            "exactStateId": "STATE-" + "A" * 20,
            "state": {"phase": "playing"},
        }
    manifest = _signed({"schemaVersion": 2, "kind": "lowering-manifest"})
    snapshot = _signed(
        {"schemaVersion": 1, "bundleHash": bundle_hash,
         "components": {"challenge": {"payload": {"tactical_detectors": copy.deepcopy(tactical_profile)} if tactical_profile else {}}}}
    )
    executor_hash = _hash("executor-v3")
    campaign = _campaign_document(executor_hash)
    domain_ir_hashes = (
        {
            name: _hash(f"domain-{name}")
            for name in ("auction", "escalation", "play", "projects", "scoring", "structural")
        }
        if compilation_receipt
        else {"rules": rules_hash}
    )
    if compilation_receipt:
        if candidate_without_normalization:
            documents["completeTrace"]["partialConstructionState"] = {"currentPlayer": "north"}
        output_hashes = {
            name: canonical_sha256(document) for name, document in documents.items()
        }
        construction_core = {
            "schemaVersion": 2,
            "kind": "ortools_research_compilation_receipt_v3",
            "backendId": "ortools_cp_sat_v3",
            "backendVersion": backend_version or "3.1.0",
            "status": execution_status or (
                "semantic_postcondition_miss"
                if candidate_without_normalization or divergent_replay_valid_candidate
                else "not_found"
            ),
            "scenarioHash": scenario["sha256"],
            "preflightHash": _hash("preflight-v3"),
            "compiledStages": [],
            "executedStages": [],
            "configurationAuthority": {
                "bundleRevisionId": configuration["bundleRevisionId"],
                "bundleHash": bundle_hash,
                "domainIrHashes": domain_ir_hashes,
                "loweringManifestHash": manifest["sha256"],
            },
            "stageReceipts": {},
            "candidateIdentity": (
                {
                    "schemaVersion": 1,
                    "origin": "search",
                    "noGoodVector": [],
                    "exactStateId": (
                        "STATE-" + "A" * 20
                        if divergent_replay_valid_candidate
                        else None
                    ),
                    "stateHash": canonical_sha256(
                        {"phase": "scored"}
                        if divergent_replay_valid_candidate
                        else {"currentPlayer": "north"}
                    ),
                }
                if candidate_without_normalization or divergent_replay_valid_candidate
                else None
            ),
            "predicateEvaluation": copy.deepcopy(predicate_evaluation),
            "semanticArtifactHashes": {
                "proofModel": output_hashes["proofModel"],
                "proofGraph": output_hashes["proofGraph"],
                "completeTrace": output_hashes["completeTrace"],
                "proofOutcome": output_hashes["proofOutcome"],
            },
        }
        documents["constructionModel"] = {
            **construction_core,
            "sha256": canonical_sha256(construction_core),
        }
    else:
        documents["constructionModel"] = {
            "schemaVersion": 1,
            "status": "not_constructed",
        }
    hashes = {name: canonical_sha256(value) for name, value in documents.items()}
    authority = ResearchAuthorityV1.build(
        backend_id="ortools_cp_sat_v3",
        backend_version=backend_version or ("3.1.0" if compilation_receipt else "3.0.0"),
        solver_version="9.15.0",
        bundle_revision_id=configuration["bundleRevisionId"],
        bundle_hash=bundle_hash,
        domain_ir_hashes=domain_ir_hashes,
        lowering_manifest_hash=manifest["sha256"],
        construction_model_hash=hashes["constructionModel"],
        proof_model_hash=hashes["proofModel"],
        graph_hash=hashes["proofGraph"],
        trace_hash=hashes["completeTrace"],
        proof_outcome_hash=hashes["proofOutcome"],
    ).to_dict()

    def reference(name: str, document: dict, media_type: str) -> dict:
        payload = _json_bytes(document)
        digest = canonical_sha256(document)
        return {
            "uri": f"cas://sha256/{digest}",
            "sha256": digest,
            "bytes": len(payload),
            "mediaType": media_type,
            "name": f"{name}.json",
        }

    artifact_references = {
        name: reference(name, document, ORTOOLS_SEMANTIC_ARTIFACT_MEDIA_TYPES[name])
        for name, document in documents.items()
    }

    def write_authority(payload: bytes, media_type: str, name: str) -> dict:
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "uri": f"cas://sha256/{digest}",
            "sha256": digest,
            "bytes": len(payload),
            "mediaType": media_type,
            "name": name,
        }

    candidate_core = (
        {
            "origin": "search",
            "state": {"currentPlayer": "north"},
            "trace": copy.deepcopy(documents["completeTrace"]),
            "stageReceipts": {},
            "exactStateId": "STATE-" + "A" * 20,
            **({"schemaVersion": 1, "noGoodVector": [], "exactStateId": None} if compilation_receipt else {}),
        }
        if candidate_without_normalization or divergent_replay_valid_candidate
        else None
    )
    if divergent_replay_valid_candidate:
        candidate_core["state"] = {"phase": "scored"}
        candidate_core["exactStateId"] = "STATE-" + "A" * 20
    candidate = (
        {**candidate_core, "sha256": canonical_sha256(candidate_core)}
        if candidate_core is not None
        else None
    )
    result = build_ortools_research_result_v3(
        run=run,
        recipe=recipe,
        executor_result={
            "schemaVersion": 1,
            "status": execution_status or (
                "semantic_postcondition_miss"
                if candidate_without_normalization or divergent_replay_valid_candidate
                else "not_found"
            ),
            "candidate": candidate,
            "predicateEvaluation": copy.deepcopy(predicate_evaluation),
            "proof": raw_proof,
            "researchAuthority": authority,
            "semanticHash": executor_hash,
            "executionReceipt": {"completedCandidates": 0},
        },
        campaign_semantic_hash=campaign["semanticHash"],
        terminal_status=terminal_status or (
            "semantic_postcondition_miss"
            if candidate_without_normalization or divergent_replay_valid_candidate
            else "not_found_within_budget"
        ),
        artifact_documents=documents,
        artifact_references=artifact_references,
        configuration_snapshot=snapshot,
        lowering_manifest=manifest,
        authority_artifact_writer=write_authority,
        runtime_compatibility_check=None,
        created_at=run["createdAt"],
    )
    return result, recipe, documents, campaign


def _v3_result() -> tuple[dict, dict]:
    result, recipe, _documents, _campaign = _v3_fixture()
    return result, recipe


@pytest.mark.parametrize("mutation", [None, "scenarioHash", "proofRequestHash", "rulesHash", "domainIrHashes", "claimSatisfied", "missing_model", "partial_certification", "false_explanation", "unowned_explanation"])
def test_counterstrategy_readback_binds_request_and_preserves_incomplete(mutation):
    result, recipe, documents, _campaign = _v3_fixture()
    authority = result["researchAuthority"]
    certificate = {
        "schemaVersion": 1, "theoremId": "SUN-ACE-LEAD-AWARDED-PROJECT-V1",
        "status": "incomplete", "facts": [],
        "scenarioHash": recipe["scenario"]["sha256"],
        "proofRequestHash": recipe["proof"]["sha256"],
        "rulesHash": recipe["configuration"]["identities"]["rules"]["sha256"],
        "domainIrHashes": {name: authority["domainIrHashes"].get(name)
                           for name in ("play", "structural", "projects", "scoring")},
        "solverVersion": authority["solverVersion"],
    }
    if mutation in {"scenarioHash", "proofRequestHash", "rulesHash"}:
        certificate[mutation] = _hash("different identity")
    elif mutation == "domainIrHashes":
        certificate[mutation] = {}
    elif mutation == "partial_certification":
        certificate["status"] = "certified"
    certificate = _signed(certificate)
    documents["proofModel"] = {"schemaVersion": 1, "kind": "native_counterstrategy_models",
                               "certificate": certificate}
    documents["proofOutcome"] = {
        "schemaVersion": 1, "kind": "native_counterstrategy_outcome",
        "certificateHash": certificate["sha256"], "claimSatisfied": None,
        "stateSpecificRootActionsEvaluated": False,
    }
    if mutation == "claimSatisfied":
        documents["proofOutcome"]["claimSatisfied"] = False
    elif mutation == "missing_model":
        documents["proofModel"] = {"kind": "proof_not_run"}
    elif mutation in {"false_explanation", "unowned_explanation"}:
        result["qualification"]["failures"].append("native_round_win_counterstrategy")
        if mutation == "unowned_explanation":
            documents["proofModel"] = {"kind": "proof_not_run"}
            documents["proofOutcome"] = {"kind": "proof_not_run"}
    for name, field in (("proofModel", "proofModelHash"), ("proofOutcome", "proofOutcomeHash")):
        authority[field] = canonical_sha256(documents[name])
    _resign_section(authority)
    if mutation is not None:
        with pytest.raises(ValueError, match="counterstrategy"):
            validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents)
    else:
        assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents) == result


@pytest.mark.parametrize("mutation", [None, "missing_pair", "actor", "unknown_field", "variables", "request", "request_utility", "decision_kind", "graph_scope", "evidence", "evidence_seed", "evidence_model", "evidence_domain", "evidence_empty", "evidence_real", "evidence_rules", "evidence_detached", "evidence_uncovered", "evidence_rehashed_score", "evidence_wrong_prefix", "evidence_reachable", "evidence_source_mismatch"])
def test_project_counterfactual_readback_preserves_owned_graph(mutation, rules, root):
    from dataclasses import replace
    from baloot_platform.research.ortools_proof import ProofGraphV1, ProofGraphNodeV1, ProofGraphEdgeV1

    result, recipe, documents, _ = _v3_fixture()
    snapshot = None
    recipe["scenario"]["decisionPoint"] = {"kind": "play" if mutation == "decision_kind" else "project"}
    recipe["scenario"]["sha256"] = canonical_sha256({key: value for key, value in recipe["scenario"].items() if key != "sha256"})
    scope = "bounded" if mutation == "graph_scope" else "exact"
    boundary = "current_raw_point_margin_v1" if scope == "bounded" else None
    graph = ProofGraphV1.build(scope=scope, boundary_evaluator=boundary, root_node_id="root", nodes=(
        ProofGraphNodeV1.decision("root", "north_south", (
            ProofGraphEdgeV1.build(0, "declare", "end"),), actor_seat="north"),
        ProofGraphNodeV1.leaf("end", 3),
    ))
    counter = ProofGraphV1.build(scope=scope, boundary_evaluator=boundary, root_node_id="root", nodes=[
        replace(node, utility=-1) if node.is_leaf else node for node in graph.nodes],
    ).to_dict()
    documents["proofGraph"] = {key: value for key, value in graph.to_dict().items() if key != "sha256"}
    model = {"graphHash": graph.sha256, "proofMode": "exact", "visibility": "complete_state",
             "utility": "awarded_point_margin", "proofRequestHash": recipe["proof"]["sha256"],
             "projectCounterfactualGraph": counter, "rootProjectConsequenceVariables": ["root_project_consequence_0"]}
    if mutation is not None and mutation.startswith("evidence"):
        from test_research_ortools_complete_state_graph import _seed
        seed = _seed(rules).to_dict()
        recipe["configuration"]["identities"]["rules"]["sha256"] = rules.sha256
        domains = {name: result["researchAuthority"]["domainIrHashes"].get(name)
                   for name in ("play", "scoring", "structural")}
        source_model = {"exactStateId": seed["exactStateId"],
            "proofStateProjectionHash": seed["proofStateProjectionHash"],
            "requestHash": recipe["proof"]["sha256"], "domainIrHashes": domains,
            "projectCounterfactualGraphHash": counter["sha256"]}
        receipt = {**source_model, "status": "complete", "exhaustionReason": None,
                   "modelHash": canonical_sha256(source_model), "graphHash": graph.sha256}
        row = {"model": source_model, "receipt": receipt, "seed": seed,
               "request": copy.deepcopy(recipe["proof"]), "graph": graph.to_dict(), "counterfactualGraph": counter}
        from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
        raw_prefix = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))["exactTracePrefix"]
        row["exactTracePrefix"] = ExactTracePrefixV1.build(raw_prefix).to_dict()
        recipe["scenario"]["origin"] = {"kind": "exact_trace_prefix", "exactTracePrefix": row["exactTracePrefix"]}
        model["projectCompilationEvidence"] = [row]
        if mutation == "evidence_seed":
            seed["rulesHash"] = _hash("substituted")
        elif mutation == "evidence_model":
            source_model["projectCounterfactualGraphHash"] = _hash("substituted")
        elif mutation == "evidence_domain":
            source_model["domainIrHashes"] = {**domains, "scoring": _hash("substituted")}
            receipt["modelHash"] = canonical_sha256(source_model)
        elif mutation == "evidence_empty":
            model["projectCompilationEvidence"] = []
        elif mutation == "evidence_rules":
            recipe["configuration"]["identities"]["rules"]["sha256"] = _hash("other-rules")
        elif mutation == "evidence_detached":
            for key in ("graph", "counterfactualGraph"):
                child = ProofGraphV1.from_dict(row[key])
                row[key] = ProofGraphV1.build(scope="exact", root_node_id="detached",
                    nodes=[replace(node, node_id="detached") if node.node_id == child.root_node_id else node
                           for node in child.nodes]).to_dict()
            source_model["projectCounterfactualGraphHash"] = row["counterfactualGraph"]["sha256"]
            receipt["modelHash"] = canonical_sha256(source_model)
            receipt["graphHash"] = row["graph"]["sha256"]
        elif mutation in {"evidence_real", "evidence_uncovered", "evidence_rehashed_score", "evidence_wrong_prefix", "evidence_reachable", "evidence_source_mismatch"}:
            from test_research_ortools_complete_state_graph import _compiler, _seed, _request
            source = _seed(rules, project_points=(20, 0))
            request = _request()
            retained_prefix = None
            if mutation != "evidence_wrong_prefix":
                from baloot_platform.research.ortools_exact_state import OrToolsExactStateNormalizerV1
                from baloot_platform.research.ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
                from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
                fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
                retained_prefix = ExactTracePrefixV1.build(fixture["exactTracePrefix"])
                normalized = OrToolsExactStateNormalizerV1(rules, native_cp_sat=True).normalize(
                    retained_prefix, decision_point={"kind": "play"}, maximum_seconds=15)
                graft = OrToolsNonPlayPlayGraphGraftV1(rules, request, native_cp_sat=True,
                    maximum_nodes=2000, maximum_depth=12, maximum_seconds=15)
                _, source = graft.native_project_play_seed(normalized, retained_prefix)
            compiled = _compiler(rules, native_conversion=True).compile(source, request,
                expected_rules_hash=rules.sha256, maximum_nodes=2000, maximum_depth=12,
                maximum_seconds=15, project_counterfactual=True)
            assert compiled.complete
            from baloot_platform.research.ortools_nonplay_play_graft import OrToolsNonPlayPlayGraphGraftV1
            actual_graph = compiled.graph
            if mutation == "evidence_rehashed_score":
                actual_graph = ProofGraphV1.build(scope="exact", root_node_id=actual_graph.root_node_id,
                    nodes=[replace(node, utility=node.utility + 1) if node.is_leaf else node for node in actual_graph.nodes],
                    tactic_context=actual_graph.tactic_context)
            child_graph = OrToolsNonPlayPlayGraphGraftV1._namespace_graph(actual_graph, "project-child")
            child_counter = OrToolsNonPlayPlayGraphGraftV1._namespace_graph(compiled.project_counterfactual_graph, "project-child")
            extra_edges = (ProofGraphEdgeV1.build(1, "pass", "uncovered"),) if mutation == "evidence_uncovered" else ()
            extra_nodes = (ProofGraphNodeV1.leaf("uncovered", 0),) if extra_edges else ()
            parent_root = ProofGraphNodeV1.decision("project-root", "north_south", (
                ProofGraphEdgeV1.build(0, "declare", child_graph.root_node_id), *extra_edges), actor_seat="north")
            graph = ProofGraphV1.build(scope="exact", root_node_id="project-root",
                nodes=(parent_root, *child_graph.nodes, *extra_nodes))
            counter = ProofGraphV1.build(scope="exact", root_node_id="project-root",
                nodes=(parent_root, *child_counter.nodes, *extra_nodes)).to_dict()
            model["rootProjectConsequenceVariables"] = [f"root_project_consequence_{index}"
                                                        for index in range(len(parent_root.edges))]
            documents["proofGraph"] = {key: value for key, value in graph.to_dict().items() if key != "sha256"}
            model["graphHash"] = graph.sha256
            model["projectCounterfactualGraph"] = counter
            actual_receipt = compiled.receipt.to_dict()
            actual_receipt.pop("elapsedSeconds")
            actual_receipt.pop("sha256")
            actual_receipt["graphHash"] = actual_graph.sha256
            model["projectCompilationEvidence"] = [{"model": compiled.model_document,
                "receipt": actual_receipt, "seed": source.to_dict(), "request": request.to_dict(),
                "graph": actual_graph.to_dict(), "counterfactualGraph": compiled.project_counterfactual_graph.to_dict()}]
            if mutation == "evidence_wrong_prefix":
                from baloot_platform.research.scenario_contracts import ExactTracePrefixV1
                fixture = json.loads((root / "examples/research-scenario-v4-ace-sacrifice-two-aces.json").read_text(encoding="utf-8"))
                model["projectCompilationEvidence"][0]["exactTracePrefix"] = ExactTracePrefixV1.build(fixture["exactTracePrefix"]).to_dict()
            if retained_prefix is not None:
                model["projectCompilationEvidence"][0]["exactTracePrefix"] = retained_prefix.to_dict()
            if "exactTracePrefix" in model["projectCompilationEvidence"][0]:
                source_prefix = copy.deepcopy(model["projectCompilationEvidence"][0]["exactTracePrefix"])
                if mutation == "evidence_source_mismatch":
                    source_prefix.pop("sha256")
                    source_prefix["matchScore"] = [1, 0]
                    source_prefix = ExactTracePrefixV1.build(source_prefix).to_dict()
                recipe["scenario"]["origin"] = {"kind": "exact_trace_prefix", "exactTracePrefix": source_prefix}
            result["researchAuthority"]["domainIrHashes"].update(actual_receipt["domainIrHashes"])
            recipe["proof"] = request.to_dict()
            recipe["configuration"]["identities"]["rules"]["sha256"] = rules.sha256
            model["proofRequestHash"] = request.sha256
            import yaml
            budgets = yaml.safe_load((root / "configs/research/proof_budgets_v1.yaml").read_text(encoding="utf-8"))
            budgets["profiles"] = {request.budget_profile: budgets["profiles"]["late_game_exact_v1"]}
            snapshot = _signed({"bundleHash": result["researchAuthority"]["bundleHash"],
                "components": {"rules": {"payload": rules.data, "sha256": rules.sha256},
                    "challenge": {"payload": {"scenarioResearchV4": {
                        "backend": "ortools_cp_sat_v3", "proofBudgets": budgets}}}}})
            result["provenance"]["resultIdentity"]["configurationSnapshotHash"] = snapshot["sha256"]
    if mutation == "missing_pair":
        del model["projectCounterfactualGraph"]
    elif mutation in {"actor", "unknown_field"}:
        if mutation == "actor":
            next(node for node in counter["nodes"] if "actorSeat" in node)["actorSeat"] = "south"
        else:
            counter["unowned"] = True
        counter["sha256"] = canonical_sha256({key: value for key, value in counter.items() if key != "sha256"})
    elif mutation == "variables":
        model["rootProjectConsequenceVariables"] = []
    elif mutation == "request":
        model["proofRequestHash"] = "0" * 64
    elif mutation == "request_utility":
        recipe["proof"]["utility"] = "round_win"
        recipe["proof"]["sha256"] = canonical_sha256({key: value for key, value in recipe["proof"].items() if key != "sha256"})
        model["proofRequestHash"] = recipe["proof"]["sha256"]
    documents["proofModel"] = model
    authority = result["researchAuthority"]
    authority["graphHash"] = canonical_sha256(documents["proofGraph"])
    authority["proofModelHash"] = canonical_sha256(model)
    _resign_section(authority)
    if mutation is None:
        assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents,
            configuration_snapshot=snapshot) == result
    else:
        expected_error = "reconstruction differs" if mutation == "evidence_rehashed_score" else "counterfactual|noncanonical|compilation|proof seed"
        if mutation in {"evidence_real", "evidence_reachable"}:
            expected_error = "project decisionPoint owner must be the current actor"
        with pytest.raises(ValueError, match=expected_error):
            validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents,
                configuration_snapshot=snapshot)


def _result_verification_service():
    from baloot_platform.research.service import ResearchFoundryServiceV1

    service = object.__new__(ResearchFoundryServiceV1)
    service._result_verification_lock = threading.RLock()
    service._result_verification_cache = {}
    return service


def test_stored_semantic_verification_coalesces_and_binds_every_input(monkeypatch):
    from baloot_platform.research import service as service_module

    service = _result_verification_service()
    value = {"resultHash": "result"}
    inputs = {
        "recipe": {"sha256": "recipe"},
        "documents": {"completeTrace": {"sha256": "trace", "actions": []}},
        "campaign_document": {"sha256": "campaign"},
        "configuration_snapshot": {"sha256": "snapshot", "components": {
            "challenge": {"payload": {"scenarioResearchV4": {
                "cacheLimits": {"maximumEntries": 8, "maximumBytes": 512}
            }}}}},
    }
    calls = []
    monkeypatch.setattr(service_module, "validate_ortools_result_artifact_documents_v3",
                        lambda result, **kwargs: calls.append(copy.deepcopy((result, kwargs))))
    ready = threading.Barrier(3)

    def read():
        ready.wait(timeout=5)
        service._verify_native_result_documents_cached(value, **inputs)

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(read) for _ in range(3)]
        for future in futures:
            future.result(timeout=5)
    assert len(calls) == 1
    # Claimed hashes deliberately stay unchanged: actual content must miss.
    for field in ("result", "recipe", "documents", "campaign_document",
                  "configuration_snapshot"):
        changed_value, changed_inputs = copy.deepcopy((value, inputs))
        if field == "result":
            changed_value["extra"] = field
        elif field == "documents":
            changed_inputs[field]["completeTrace"]["actions"].append("changed")
        else:
            changed_inputs[field]["extra"] = field
        service._verify_native_result_documents_cached(changed_value, **changed_inputs)
    assert len(calls) == 6
    fresh = _result_verification_service()
    fresh._verify_native_result_documents_cached(value, **inputs)
    assert len(calls) == 7


@pytest.mark.real_clock
def test_stored_semantic_verification_hit_does_not_wait_for_unrelated_proof(monkeypatch):
    from baloot_platform.research import service as service_module

    service = _result_verification_service()
    inputs = {"recipe": {}, "documents": {}, "campaign_document": None,
              "configuration_snapshot": {"components": {"challenge": {"payload": {
                  "scenarioResearchV4": {"cacheLimits": {
                      "maximumEntries": 2, "maximumBytes": 128}}}}}}}
    started, release = threading.Event(), threading.Event()
    calls = []

    def verify(result, **_kwargs):
        calls.append(result["result"])
        if result["result"] == "unrelated":
            started.set()
            assert release.wait(timeout=10)

    monkeypatch.setattr(service_module, "validate_ortools_result_artifact_documents_v3", verify)
    service._verify_native_result_documents_cached({"result": "verified"}, **inputs)
    with ThreadPoolExecutor(max_workers=2) as pool:
        cold = pool.submit(service._verify_native_result_documents_cached,
                           {"result": "unrelated"}, **inputs)
        try:
            assert started.wait(timeout=5)
            warm = pool.submit(service._verify_native_result_documents_cached,
                               {"result": "verified"}, **inputs)
            warm.result(timeout=2)
            assert not cold.done()
        finally:
            release.set()
        cold.result(timeout=5)
    assert calls == ["verified", "unrelated"]


def test_stored_semantic_verification_does_not_cache_failure_and_obeys_limits(monkeypatch):
    from baloot_platform.research import service as service_module

    service = _result_verification_service()
    inputs = {"recipe": {}, "documents": {}, "campaign_document": None,
              "configuration_snapshot": {"components": {"challenge": {"payload": {
                  "scenarioResearchV4": {"cacheLimits": {
                      "maximumEntries": 1, "maximumBytes": 64}}}}}}}
    calls = []

    def verify(result, **_kwargs):
        calls.append(result)
        if len(calls) == 1:
            raise ValueError("invalid proof")

    monkeypatch.setattr(service_module, "validate_ortools_result_artifact_documents_v3", verify)
    with pytest.raises(ValueError, match="invalid proof"):
        service._verify_native_result_documents_cached({"result": 1}, **inputs)
    assert not service._result_verification_cache
    for result in (1, 1, 2, 1):
        service._verify_native_result_documents_cached({"result": result}, **inputs)
        assert len(service._result_verification_cache) == 1
    assert len(calls) == 4
    inputs["configuration_snapshot"]["components"] = {}
    for _ in range(2):
        service._verify_native_result_documents_cached({"result": 1}, **inputs)
    assert len(calls) == 6  # Historical snapshots without limits remain uncached.


def test_cached_result_read_still_checks_cas_ownership_and_run(monkeypatch):
    import sys
    from baloot_platform.research import service as service_module

    payloads, captured = {}, {}
    builder = build_ortools_research_result_v3

    def capture(**kwargs):
        snapshot = kwargs["configuration_snapshot"]
        snapshot["components"] = {"challenge": {"payload": {
            "scenarioResearchV4": {"cacheLimits": {
                "maximumEntries": 8, "maximumBytes": 512}}}}}
        _resign_section(snapshot)
        writer = kwargs["authority_artifact_writer"]

        def write(payload, media_type, name):
            reference = writer(payload, media_type, name)
            payloads[reference["uri"]] = payload
            return reference

        kwargs["authority_artifact_writer"] = write
        captured.update(kwargs)
        return builder(**kwargs)

    monkeypatch.setattr(sys.modules[__name__], "build_ortools_research_result_v3", capture)
    result, recipe, documents, _campaign = _v3_fixture()
    for name, document in documents.items():
        payloads[result["artifacts"][name]["uri"]] = _json_bytes(document)
    service = _result_verification_service()
    reads = []

    def load(uri):
        reads.append(uri)
        return payloads[uri]

    service.artifact_store = SimpleNamespace(get=load)
    # This fixture is an intentionally minimal historical recipe; retain real
    # result/CAS/run validators while bypassing unrelated authoring requirements.
    monkeypatch.setattr(service_module.ResearchRecipeV4, "build",
                        lambda _value: SimpleNamespace(to_dict=lambda: recipe))
    checks = []
    monkeypatch.setattr(service_module, "validate_ortools_result_artifact_documents_v3",
                        lambda *_args, **_kwargs: checks.append(True))
    run = captured["run"]
    for _ in range(2):
        assert service._validate_stored_scenario_result_v4(
            result, run=run, recipe_value=recipe) == result
    assert checks == [True]
    assert len(reads) == 2 * (len(result["artifacts"]) + len(result["authorityArtifacts"]))
    trace_uri = result["artifacts"]["completeTrace"]["uri"]
    payloads[trace_uri] += b" "
    with pytest.raises(ValueError, match="integrity check failed"):
        service._validate_stored_scenario_result_v4(result, run=run, recipe_value=recipe)
    payloads[trace_uri] = _json_bytes(documents["completeTrace"])
    substituted = copy.deepcopy(result)
    substituted["artifacts"]["completeTrace"]["ownerHash"] = _hash("wrong owner")
    with pytest.raises((ValueError, RuntimeError), match="substitution detected"):
        service._validate_stored_scenario_result_v4(substituted, run=run, recipe_value=recipe)
    with pytest.raises(RuntimeError, match="result owner identity mismatch"):
        service._validate_stored_scenario_result_v4(
            result, run={**run, "runHash": _hash("wrong run")}, recipe_value=recipe)
    for authority in (None, {**run["configurationAuthority"], "semanticBundleHash": _hash("wrong bundle")}):
        wrong_run = {**run, "configurationAuthority": authority}
        with pytest.raises(RuntimeError, match="bundle authority mismatch"):
            service._validate_stored_scenario_result_v4(result, run=wrong_run, recipe_value=recipe)
    assert checks == [True]


def test_rejected_candidate_without_exact_normalization_does_not_claim_exact_state() -> None:
    result, recipe, _documents, _campaign = _v3_fixture(
        candidate_without_normalization=True
    )

    assert result["scenarioProjection"]["exactStateId"] is None
    assert result["scenarioProjection"]["exactStateAvailable"] is False
    assert result["provenance"]["resultIdentity"]["exactStateId"] is None
    assert validate_ortools_result_sections_v1(result, recipe=recipe) == result


def test_immediate_storage_quality_ownership_is_private_and_falls_back(monkeypatch, rules):
    import inspect
    import sys
    from baloot_platform.research import result_authority as authority_module

    captured = []
    original_builder = build_ortools_research_result_v3
    def capture_inputs(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return original_builder(**kwargs)
    monkeypatch.setattr(sys.modules[__name__], "build_ortools_research_result_v3", capture_inputs)
    _v3_fixture()
    arguments = captured[0]
    arguments["recipe"]["playingQuality"] = {"schemaVersion": 1, "mode": "exact", "decisions": {
        kind: {"utility": "awarded_point_margin", "budgetProfile": "test"}
        for kind in ("bid", "escalation", "project", "play")}}
    arguments["artifact_documents"]["completeTrace"]["playingQualityEvidence"] = {
        "receipt": {"status": "complete"}}
    snapshot = arguments["configuration_snapshot"]
    snapshot["components"] = {"rules": {"payload": rules.data, "sha256": rules.sha256},
        "challenge": {"payload": {"scenarioResearchV4": {
            "backend": "ortools_cp_sat_v3", "proofBudgets": {"profiles": {"test": {"maximumNodes": 17}}},
            "qualityExecution": {"schemaVersion": 1, "maximumWallSeconds": 19,
                "maximumEvidenceBytes": 23, "maximumEscalationDecisions": 7}}}}}
    snapshot["sha256"] = canonical_sha256({k: v for k, v in snapshot.items() if k != "sha256"})
    owned = frozenset({_hash("owned-proof")})
    with pytest.raises(ValueError, match="completeTrace semantic artifact preimage mismatch"):
        authority_module._build_ortools_research_result_v3(**arguments, verified_artifacts=owned)
    document = arguments["artifact_documents"]["completeTrace"]
    digest = canonical_sha256(document)
    authority = arguments["executor_result"]["researchAuthority"]
    authority["traceHash"] = digest
    authority["sha256"] = canonical_sha256({k: v for k, v in authority.items() if k != "sha256"})
    arguments["artifact_references"]["completeTrace"].update(
        uri=f"cas://sha256/{digest}", sha256=digest, bytes=len(_json_bytes(document)))
    calls = []
    def check_quality(_trace, **kwargs):
        calls.append(kwargs.get("verified_artifacts", frozenset()))
        assert kwargs["rules"].sha256 == rules.sha256
        assert kwargs["required_execution"].maximum_wall_seconds == 19
        assert kwargs["required_proof_profiles"] == {"test": {"maximumNodes": 17}}
        raise ValueError("quality handoff checked")
    monkeypatch.setattr(authority_module, "_validate_trace_quality_evidence_v1", check_quality)
    monkeypatch.setattr(authority_module, "validate_trace_quality_evidence_v1", check_quality)
    for builder, extra in ((authority_module._build_ortools_research_result_v3,
                            {"verified_artifacts": owned}),
                           (authority_module._build_ortools_research_result_v3, {}),
                           (authority_module.build_ortools_research_result_v3, {})):
        with pytest.raises(ValueError, match="quality handoff checked"):
            builder(**arguments, **extra)
    assert calls == [owned, frozenset(), frozenset()]
    assert "verified_artifacts" not in inspect.signature(original_builder).parameters


def test_rejected_replay_valid_candidate_state_mismatch_fails_before_storage() -> None:
    with pytest.raises(ValueError, match="completeTrace exact state does not match"):
        _v3_fixture(
            compilation_receipt=True,
            divergent_replay_valid_candidate=True,
        )


@pytest.mark.parametrize("tampered", [False, True])
def test_partial_compilation_state_is_owned_without_exactness(tampered):
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True, candidate_without_normalization=True)
    if tampered:
        construction = documents["constructionModel"]
        construction["candidateIdentity"]["stateHash"] = _hash("substituted partial state")
        _resign_section(construction)
        result["researchAuthority"]["constructionModelHash"] = canonical_sha256(construction)
        _resign_section(result["researchAuthority"])
        with pytest.raises(RuntimeError, match="exact-state identity mismatch"):
            validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents, campaign_document=campaign)
    else:
        assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe, documents=documents,
            campaign_document=campaign) == result
        assert result["scenarioProjection"]["exactStateAvailable"] is False


@pytest.mark.parametrize("has_candidate", [False, True])
def test_construction_timeout_readback_requires_no_candidate(has_candidate):
    result, recipe, documents, campaign = _v3_fixture(
        compilation_receipt=True,
        candidate_without_normalization=has_candidate,
        terminal_status="not_found_within_budget",
    )
    construction = documents["constructionModel"]
    construction["status"] = "incomplete"
    _resign_section(construction)
    result["researchAuthority"]["constructionModelHash"] = canonical_sha256(construction)
    _resign_section(result["researchAuthority"])
    if has_candidate:
        with pytest.raises(RuntimeError, match="compilation receipt identity mismatch"):
            validate_ortools_result_artifact_documents_v3(
                result, recipe=recipe, documents=documents, campaign_document=campaign,
            )
    else:
        assert validate_ortools_result_artifact_documents_v3(
            result, recipe=recipe, documents=documents, campaign_document=campaign,
        ) == result


def _resign_v3_result(result: dict) -> None:
    identity = result["provenance"]["resultIdentity"]
    result_hash = canonical_sha256(identity)
    result["resultHash"] = result_hash
    receipt = result["executionReceipt"]
    receipt["semanticResultHash"] = result_hash
    receipt["sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    for reference in result["authorityArtifacts"].values():
        reference["ownerHash"] = result_hash
    for name, document in (
        ("resultIdentity", identity),
        ("executionReceipt", receipt),
    ):
        reference = result["authorityArtifacts"][name]
        payload = _json_bytes(document)
        digest = hashlib.sha256(payload).hexdigest()
        reference.update(
            uri=f"cas://sha256/{digest}",
            sha256=digest,
            bytes=len(payload),
            name=f"{digest}.json",
        )


def _stage_receipt_v2() -> dict:
    model_hash = _hash("deal-model-v2")
    evidence_hash = _hash("deal-evidence-v2")
    return _signed(
        {
            "schemaVersion": 2,
            "stage": "deal",
            "modelHash": model_hash,
            "evidenceHash": evidence_hash,
            "semanticHash": _hash("deal-semantic-v2"),
            "solverStatus": "infeasible",
            "variableCount": 3,
            "constraintCount": 4,
            "namedAssumptions": [
                {
                    "assumptionId": "ASSUME-DEAL",
                    "literalName": "assume_deal",
                    "literalIndex": 2,
                    "predicateIds": ["PRED-DEAL"],
                }
            ],
            "infeasibleCore": ["ASSUME-DEAL"],
            "evidenceBindings": [
                {
                    "predicateId": "PRED-DEAL",
                    "sourceEvidenceHash": evidence_hash,
                    "loweringHash": _hash("deal-lowering-v2"),
                    "modelHash": model_hash,
                    "literalName": "pred_deal",
                    "literalIndex": 1,
                    "evidenceHash": None,
                }
            ],
            "elapsedSeconds": 0.01,
            "branches": 0,
            "conflicts": 0,
        }
    )


def _resign_section(document: dict) -> None:
    document["sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "sha256"}
    )


def _attach_predicate_evaluation(result: dict, *, kind: str | None = None) -> None:
    evidence = _signed(
        {
            "schemaVersion": 1,
            "predicateId": "PRED-RECEIPT",
            **({"kind": kind} if kind is not None else {}),
            "satisfied": False,
        }
    )
    predicate = _signed(
        {
            "schemaVersion": 1,
            "status": "target_miss",
            "satisfied": False,
            "evidence": [evidence],
        }
    )
    result["predicateEvaluation"] = predicate
    identity = result["provenance"]["resultIdentity"]
    identity["predicateEvaluationHash"] = predicate["sha256"]
    satisfaction = result["scenarioSatisfaction"]
    satisfaction["predicateEvaluationHash"] = predicate["sha256"]
    satisfaction["predicates"] = [evidence]
    _resign_section(satisfaction)
    identity["scenarioSatisfactionHash"] = satisfaction["sha256"]
    _resign_v3_result(result)


def test_authority_build_requires_an_explicit_backend() -> None:
    kwargs = {
        key: value
        for key, value in {
            "backend_version": "3.0.0",
            "solver_version": "9.15.0",
            "bundle_revision_id": "CBR-EXPLICIT",
            "bundle_hash": _hash("bundle"),
            "domain_ir_hashes": {"rules": _hash("rules")},
            "lowering_manifest_hash": _hash("lowering"),
            "construction_model_hash": _hash("construction"),
            "proof_model_hash": _hash("proof-model"),
            "graph_hash": _hash("graph"),
            "trace_hash": _hash("trace"),
            "proof_outcome_hash": _hash("proof"),
        }.items()
    }
    with pytest.raises(TypeError, match="backend_id"):
        ResearchAuthorityV1.build(**kwargs)


def test_v2_identity_rejects_explicit_v3_backend_authority() -> None:
    with pytest.raises(ValueError, match="explicit V2 backend"):
        build_ortools_result_identity_v2(
            {},
            _authority("ortools_cp_sat_v3"),
            domain_authority_hash=_hash("domain-authority"),
            executor_semantic_hash=_hash("executor"),
            candidate_hash=_hash("candidate"),
            predicate_evaluation_hash=_hash("predicate"),
        )


@pytest.mark.parametrize("identity_kind", (None, "unknown_identity_v1"))
def test_result_read_never_falls_back_when_identity_kind_is_not_explicit(
    identity_kind: str | None,
) -> None:
    identity = {} if identity_kind is None else {"kind": identity_kind}
    with pytest.raises(RuntimeError, match="identity kind is missing or unsupported"):
        validate_ortools_result_sections_v1(
            {"provenance": {"resultIdentity": identity}}, recipe={}
        )


@pytest.mark.parametrize(
    "field",
    (
        "backendId",
        "bundleRevisionId",
        "bundleHash",
        "constructionModelHash",
        "proofModelHash",
        "graphHash",
        "traceHash",
        "proofOutcomeHash",
    ),
)
def test_v3_read_rejects_coherently_resigned_authority_identity_mismatch(
    field: str,
) -> None:
    result, recipe = _v3_result()
    result["provenance"]["resultIdentity"][field] = (
        "ortools_cp_sat_v2" if field == "backendId" else _hash(f"other-{field}")
    )
    if field == "bundleRevisionId":
        result["provenance"]["resultIdentity"][field] = "CBR-OTHER"
    _resign_v3_result(result)

    with pytest.raises(RuntimeError, match="semantic identity mismatch"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


@pytest.mark.parametrize("field", ("bundleHash", "graphHash", "modelHash"))
def test_v3_read_rejects_coherently_resigned_proof_identity_mismatch(
    field: str,
) -> None:
    result, recipe = _v3_result()
    proof = result["proofOutcome"]
    proof["identities"][field] = _hash(f"other-proof-{field}")
    proof["sha256"] = canonical_sha256(
        {key: value for key, value in proof.items() if key != "sha256"}
    )
    result["provenance"]["resultIdentity"]["proofOutcomeReceiptHash"] = proof["sha256"]
    _resign_v3_result(result)

    with pytest.raises(RuntimeError, match="scenario authority mismatch"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


def test_v3_stage_receipt_v2_validates_complete_solver_evidence() -> None:
    receipt = _stage_receipt_v2()

    assert validate_ortools_stage_receipt_v2(receipt) == receipt


@pytest.mark.parametrize(
    "case",
    (
        "missing_field",
        "unknown_field",
        "invalid_model_hash",
        "unsupported_solver_status",
        "negative_variable_count",
        "duplicate_assumption",
        "unknown_core_assumption",
        "binding_model_substitution",
        "binding_evidence_substitution",
        "negative_elapsed_time",
        "tampered_integrity",
    ),
)
def test_v3_stage_receipt_v2_rejects_tamper_omission_and_substitution(
    case: str,
) -> None:
    receipt = _stage_receipt_v2()
    if case == "missing_field":
        receipt.pop("constraintCount")
    elif case == "unknown_field":
        receipt["solverWallTime"] = 0.01
    elif case == "invalid_model_hash":
        receipt["modelHash"] = "not-a-hash"
    elif case == "unsupported_solver_status":
        receipt["solverStatus"] = "satisfied"
    elif case == "negative_variable_count":
        receipt["variableCount"] = -1
    elif case == "duplicate_assumption":
        receipt["namedAssumptions"].append(
            copy.deepcopy(receipt["namedAssumptions"][0])
        )
    elif case == "unknown_core_assumption":
        receipt["infeasibleCore"] = ["ASSUME-OTHER"]
    elif case == "binding_model_substitution":
        receipt["evidenceBindings"][0]["modelHash"] = _hash("other-model")
    elif case == "binding_evidence_substitution":
        receipt["evidenceBindings"][0]["sourceEvidenceHash"] = _hash(
            "other-evidence"
        )
    elif case == "negative_elapsed_time":
        receipt["elapsedSeconds"] = -0.01
    elif case == "tampered_integrity":
        receipt["branches"] = 1
    if case != "tampered_integrity":
        _resign_section(receipt)

    with pytest.raises((TypeError, ValueError)):
        validate_ortools_stage_receipt_v2(receipt)


def test_v3_1_compilation_receipt_closes_all_semantic_preimages() -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)

    assert validate_ortools_result_sections_v1(result, recipe=recipe) == result
    assert (
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=campaign,
        )
        == result
    )


def _tactic_projection_fixture(*, wrapped: bool, mutation: str | None = None, truth: bool = True):
    """Audit-result plumbing only, not a qualified game or strategic proof."""
    native = _signed({
        "schemaVersion": 1, "predicateId": "PRED-TACTIC", "kind": "tactic.verified",
        "satisfied": truth,
        "evidenceExtractor": "cp_sat.registered_tactic_verified_evidence_v1",
        "loweringHash": _hash("tactic-lowering"),
        "observed": {"detectorId": "only_winning_move", "applicable": truth, "verified": truth},
    })
    owned_native = copy.deepcopy(native)
    if mutation in {"predicateId", "kind", "evidenceExtractor"}:
        native[mutation] = "wrong-owner"
    elif mutation == "schemaVersion":
        native[mutation] = True
    elif mutation == "satisfied":
        native[mutation] = 1
    elif mutation == "detector":
        native["observed"]["detectorId"] = "debate"
    if mutation != "checksum":
        _resign_section(native)
    else:
        native["observed"]["verified"] = False
    evidence = _signed({
        "schemaVersion": 1, "predicateId": "PRED-TACTIC", "kind": "tactic.verified",
        "satisfied": truth, "evidenceExtractor": "cp_sat.native_proof_same_solve_v1",
        "loweringHash": _hash("same-solve-lowering"), "observed": native,
    }) if wrapped else native
    if mutation == "outer_truth":
        evidence["satisfied"] = not truth
        _resign_section(evidence)
    evaluation = _signed({"schemaVersion": 1, "status": "satisfied", "satisfied": True,
                          "evidence": [evidence]})
    proof = _signed({"completionStatus": "exact_complete", "solverStatus": "optimal",
        "claimSatisfied": False, "nativeQualification": {
        "PRED-TACTIC": owned_native if mutation == "proof" else copy.deepcopy(native),
    }})
    if mutation == "proof":
        proof["nativeQualification"]["PRED-TACTIC"]["observed"]["verified"] = not truth
        _resign_section(proof["nativeQualification"]["PRED-TACTIC"])
        _resign_section(proof)
    return _v3_fixture(compilation_receipt=True, raw_proof=proof,
        predicate_expression={"op": "all", "children": [{
            "predicateId": "PRED-TACTIC", "kind": "tactic.verified", "version": 1,
            "hardness": "report", "arguments": {"kind": "only_winning_move",
                "minimumScore": 0, "minimumEvidence": 1}}]},
        predicate_evaluation=evaluation)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("truth", [False, True])
def test_native_tactical_tags_survive_creation_and_artifact_read(wrapped, truth):
    result, recipe, documents, campaign = _tactic_projection_fixture(wrapped=wrapped, truth=truth)
    assert result["tacticalTags"] == ["only_winning_move"]
    assert result["verifiedTacticalTags"] == (["only_winning_move"] if truth else [])
    assert result["disposition"] == "rejected"  # No qualification is supplied by this fixture.
    assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
        documents=documents, campaign_document=campaign) == result


@pytest.mark.parametrize("truth,mutation,reason", [
    (True, "checksum", "inner checksum"),
    (True, "schemaVersion", "ownership mismatch"),
    (True, "predicateId", "ownership mismatch"),
    (True, "kind", "ownership mismatch"),
    (True, "satisfied", "ownership mismatch"),
    (True, "evidenceExtractor", "ownership mismatch"),
    (True, "detector", "does not match the scenario"),
    (True, "proof", "does not match the proof"),
    (True, "outer_truth", "ownership mismatch"),
    (False, "outer_truth", "ownership mismatch"),
    (False, "proof", "does not match the proof"),
])
def test_same_solve_tactical_tags_reject_foreign_or_corrupt_inner_evidence(truth, mutation, reason):
    with pytest.raises(ValueError, match=reason):
        _tactic_projection_fixture(wrapped=True, mutation=mutation, truth=truth)


@pytest.mark.parametrize("truth", [False, True])
def test_same_solve_tactical_tag_read_checks_false_and_true_wrappers(truth):
    result, recipe, documents, campaign = _tactic_projection_fixture(wrapped=True, truth=truth)
    evaluation = result["predicateEvaluation"]
    evidence = evaluation["evidence"][0]
    evidence["satisfied"] = not truth
    _resign_section(evidence)
    _resign_section(evaluation)
    with pytest.raises(ValueError, match="inner ownership mismatch"):
        validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign)


def test_native_tactical_tag_read_rejects_rehashed_unproved_tags():
    result, recipe, documents, campaign = _tactic_projection_fixture(wrapped=False)
    result["verifiedTacticalTags"] = ["debate"]
    result["provenance"]["resultIdentity"]["verdictHash"] = canonical_sha256({
        key: result[key] for key in ("truthLabel", "confidence", "score", "disposition",
            "classification", "criticalDecisionIds", "interestingSeats", "tacticalTags",
            "verifiedTacticalTags", "decisionCount")})
    _resign_v3_result(result)
    with pytest.raises(RuntimeError, match="tactical tag projection"):
        validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign)


@pytest.mark.parametrize("backend_version", ["3.1.136", "3.1.137"])
@pytest.mark.parametrize("solved_roots", [False, True])
def test_incomplete_receipt_never_invents_new_root_valuations(
    backend_version, solved_roots,
):
    roots = [{"action": "CA", "value": -8, "satisfies": False}] if solved_roots else []
    raw_proof = {
        "completionStatus": "proof_incomplete", "claimSatisfied": False,
        "rootActions": roots, "winningRootActionCount": None,
        "winningRootActionLowerBound": 0, "winningRootActionUpperBound": None,
    }
    result = _terminal_proof_source_v3(
        raw_proof=raw_proof, terminal_status="proof_incomplete",
        proof_request={}, exact_normalization={"legalActions": [{"code": "CA"}, {"code": "C7"}]},
        backend_version=backend_version,
    )
    assert raw_proof["rootActions"] == roots
    assert result["winningRootActionCount"] is None
    if solved_roots:
        assert result["rootActions"] == roots
    elif backend_version == "3.1.136":
        assert result["rootActions"] == [
            {"action": action, "value": 0, "satisfies": False} for action in ("CA", "C7")
        ]
    else:
        assert result["rootActions"] == []
        assert result["winningRootActionLowerBound"] == 0
        assert result["winningRootActionUpperBound"] == 2


@pytest.mark.parametrize("terminal_status", ["proof_failed", "proof_incomplete"])
@pytest.mark.parametrize("mutation", [None, "root_value", "selected_action", "winning_count"])
def test_rejected_v3_receipt_is_derived_from_owned_proof(terminal_status, mutation):
    raw_proof = _signed({
        "completionStatus": "exact_complete", "claimSatisfied": False,
        "rootActions": [{"action": "CA", "value": -8, "satisfies": False},
                        {"action": "C7", "value": -20, "satisfies": False}],
        "selectedAction": "CA", "winningRootActionCount": 0,
    })
    result, recipe, documents, campaign = _v3_fixture(
        compilation_receipt=True, raw_proof=raw_proof, terminal_status=terminal_status)
    assert result["proofOutcome"]["completionStatus"] == terminal_status
    assert result["proofOutcome"]["rootActions"] == raw_proof["rootActions"]
    if mutation is not None:
        receipt = result["proofOutcome"]
        if mutation == "root_value":
            receipt["rootActions"][0]["value"] = -7
        elif mutation == "selected_action":
            receipt["selectedAction"] = "C7"
        else:
            receipt.update(winningRootActionCount=1, winningRootActionLowerBound=1,
                           winningRootActionUpperBound=1)
            receipt["rootActions"][0]["satisfies"] = True
            projection = result["scenarioProjection"]
            projection.update(winningRootActionCount=1, winningRootActionLowerBound=1,
                              winningRootActionUpperBound=1)
            _resign_section(projection)
            result["provenance"]["resultIdentity"]["scenarioProjectionHash"] = projection["sha256"]
        _resign_section(receipt)
        result["provenance"]["resultIdentity"]["proofOutcomeReceiptHash"] = receipt["sha256"]
        _resign_v3_result(result)
        assert validate_ortools_result_sections_v1(result, recipe=recipe) == result
        with pytest.raises(RuntimeError, match="proof receipt substitution"):
            validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
                documents=documents, campaign_document=campaign)
    else:
        assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign) == result


@pytest.mark.parametrize("terminal_status", ["proof_failed", "proof_incomplete"])
@pytest.mark.parametrize("missing_proof", [False, True])
def test_rejected_v3_creation_rejects_proof_artifact_split(monkeypatch, terminal_status, missing_proof):
    import sys

    original_builder = build_ortools_research_result_v3
    def substitute_proof(**kwargs):
        supplied = _signed({
            "completionStatus": terminal_status, "claimSatisfied": False,
            "rootActions": [{"action": "CA", "value": -8, "satisfies": False}],
            "selectedAction": "CA", "winningRootActionCount": 0,
        })
        kwargs["executor_result"]["proof"] = None if missing_proof else supplied
        return original_builder(**kwargs)
    monkeypatch.setattr(sys.modules[__name__], "build_ortools_research_result_v3", substitute_proof)
    with pytest.raises(ValueError, match="proofOutcome artifact does not match V3 proof"):
        _v3_fixture(compilation_receipt=True, terminal_status=terminal_status,
                    raw_proof=_signed({"completionStatus": terminal_status,
                                       "claimSatisfied": False}) if missing_proof else None)


def test_stored_trace_rejects_quality_evidence_even_with_rehashed_preimage() -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)
    trace = documents["completeTrace"]
    trace["playingQualityEvidence"] = {}
    trace["exactNormalization"] = {"state": {"rulesHash": _hash("rules-v3")}}
    authority = result["researchAuthority"]
    authority["traceHash"] = canonical_sha256(trace)
    _resign_section(authority)
    with pytest.raises(ValueError, match="trace quality evidence shape mismatch"):
        validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign)


def test_v3_0_read_remains_compatible_without_compilation_receipt() -> None:
    result, recipe, documents, _campaign = _v3_fixture()

    assert (
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=None,
        )
        == result
    )


@pytest.mark.parametrize("case", ["missing_evidence", "missing_snapshot", "tampered_snapshot", "wrong_bundle", "wrong_backend", "valid_snapshot"])
def test_stored_qualified_trace_requires_recipe_quality_evidence(case, monkeypatch, rules) -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)
    result["disposition"] = "qualified"
    recipe["playingQuality"] = {"schemaVersion": 1, "mode": "exact", "decisions": {
        kind: {"utility": "awarded_point_margin", "budgetProfile": "test"}
        for kind in ("bid", "escalation", "project", "play")}}
    trace = documents["completeTrace"]
    trace["exactNormalization"] = {"state": {"rulesHash": rules.sha256}}
    snapshot = None
    expected = "requires quality evidence"
    if case != "missing_evidence":
        trace["playingQualityEvidence"] = {}
        expected = "configuration snapshot authority mismatch"
        if case != "missing_snapshot":
            snapshot = _signed({"bundleHash": result["researchAuthority"]["bundleHash"],
                "components": {"rules": {"payload": rules.data, "sha256": rules.sha256},
                    "challenge": {"payload": {"scenarioResearchV4": {"backend": "wrong"}}}}})
            if case == "wrong_bundle":
                snapshot["bundleHash"] = _hash("other-bundle")
                _resign_section(snapshot)
            result["provenance"]["resultIdentity"]["configurationSnapshotHash"] = snapshot["sha256"]
            if case == "tampered_snapshot":
                snapshot["extra"] = True
            if case == "wrong_backend":
                expected = "configuration backend mismatch"
            if case == "valid_snapshot":
                settings = snapshot["components"]["challenge"]["payload"]["scenarioResearchV4"]
                settings.update(backend=result["researchAuthority"]["backendId"],
                    proofBudgets={"profiles": {"test": {"maximumNodes": 17}}},
                    qualityExecution={"schemaVersion": 1, "maximumWallSeconds": 19,
                                      "maximumEvidenceBytes": 23, "maximumEscalationDecisions": 7})
                _resign_section(snapshot)
                result["provenance"]["resultIdentity"]["configurationSnapshotHash"] = snapshot["sha256"]
                def check_frozen_arguments(actual_trace, **kwargs):
                    assert actual_trace == trace
                    assert kwargs["rules"].sha256 == rules.sha256
                    assert kwargs["rules"].data == rules.data
                    assert kwargs["required_policy"].to_dict()["decisions"] == recipe["playingQuality"]["decisions"]
                    assert kwargs["required_proof_profiles"] == settings["proofBudgets"]["profiles"]
                    assert kwargs["required_execution"].maximum_wall_seconds == 19
                    assert kwargs["required_execution"].maximum_evidence_bytes == 23
                    assert kwargs["required_execution"].maximum_escalation_decisions == 7
                    raise ValueError("verified frozen argument handoff")
                monkeypatch.setattr("baloot_platform.research.result_authority.validate_trace_quality_evidence_v1",
                                    check_frozen_arguments)
                expected = "verified frozen argument handoff"
    result["researchAuthority"]["traceHash"] = canonical_sha256(trace)
    _resign_section(result["researchAuthority"])
    with pytest.raises(ValueError, match=expected):
        validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign, configuration_snapshot=snapshot)


@pytest.mark.parametrize("omission", ("campaign", "semantic_artifact"))
def test_v3_1_read_rejects_missing_semantic_preimages(omission: str) -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)
    if omission == "campaign":
        campaign = None
    else:
        documents.pop("proofGraph")

    with pytest.raises(RuntimeError):
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=campaign,
        )


def test_v3_1_read_rejects_construction_artifact_tamper() -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)
    documents["constructionModel"]["preflightHash"] = _hash("substituted-preflight")

    with pytest.raises(RuntimeError, match="constructionModel preimage mismatch"):
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=campaign,
        )


def test_v3_1_read_rejects_campaign_receipt_substitution() -> None:
    result, recipe, documents, _campaign = _v3_fixture(compilation_receipt=True)
    substituted_campaign = _campaign_document(_hash("substituted-executor"))

    with pytest.raises(RuntimeError, match="campaign/result semantic binding mismatch"):
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=substituted_campaign,
        )


@pytest.mark.parametrize(
    "field",
    ("candidateHash", "exactStateHash", "campaignSemanticHash"),
)
def test_v3_1_read_rejects_orphaned_or_substituted_result_identity(
    field: str,
) -> None:
    result, recipe, documents, campaign = _v3_fixture(compilation_receipt=True)
    identity = result["provenance"]["resultIdentity"]
    identity[field] = _hash(f"substituted-{field}")
    if field == "campaignSemanticHash":
        result["provenance"]["campaignSemanticHash"] = identity[field]
    _resign_v3_result(result)
    assert validate_ortools_result_sections_v1(result, recipe=recipe) == result

    with pytest.raises(RuntimeError):
        validate_ortools_result_artifact_documents_v3(
            result,
            recipe=recipe,
            documents=documents,
            campaign_document=campaign,
        )


def test_v3_read_recomputes_predicate_evaluation_hashes() -> None:
    result, recipe, _, _ = _v3_fixture(predicate_expression={"op": "all", "children": [{
        "predicateId": "PRED-RECEIPT", "kind": "auction.contract", "version": 1,
        "hardness": "hard", "arguments": {"contract": "sun"}}]})
    _attach_predicate_evaluation(result, kind="auction.contract")
    assert validate_ortools_result_sections_v1(result, recipe=recipe) == result

    result["predicateEvaluation"]["satisfied"] = True

    with pytest.raises(RuntimeError, match="semantic preimage is invalid"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


def test_v3_read_rejects_correctly_hashed_unowned_predicate_evidence() -> None:
    result, recipe = _v3_result()
    _attach_predicate_evaluation(result)
    with pytest.raises(RuntimeError, match="predicate evidence"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


@pytest.mark.parametrize("operator,truths,bounds,expected", [
    ("all", [True, False], {}, False),
    ("any", [True, False], {}, True),
    ("any", [False, False], {}, False),
    ("not", [False], {}, True),
    ("not", [True], {}, False),
    ("count", [True, False], {"minimum": 1, "maximum": 1}, True),
    ("count", [True, True], {"minimum": 1, "maximum": 1}, False),
])
def test_v3_predicate_readback_recomputes_hard_expression(operator, truths, bounds, expected):
    from baloot_platform.research.result_authority import _validate_predicate_evaluation_v1

    children = [{"predicateId": f"PRED-RECEIPT-{index}", "kind": "auction.contract",
        "version": 1, "hardness": "hard", "arguments": {"contract": "sun"}}
        for index in range(len(truths))]
    scenario = {"constraints": {"op": operator, "children": children, **bounds},
        "outcome": {"op": "all", "children": []}}
    value = _signed({"schemaVersion": 1, "status": "satisfied", "satisfied": True,
        "evidence": [_signed({"predicateId": child["predicateId"], "kind": child["kind"],
            "satisfied": truth}) for child, truth in zip(children, truths)]})
    if expected:
        assert _validate_predicate_evaluation_v1(value, expected_hash=value["sha256"], scenario=scenario) == value
    else:
        with pytest.raises(ValueError, match="contradicts the hard expression"):
            _validate_predicate_evaluation_v1(value, expected_hash=value["sha256"], scenario=scenario)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "kind", "coercion", "projection"])
def test_v3_read_rejects_resigned_predicate_coverage_and_projection(mutation):
    result, recipe, _, _ = _v3_fixture(predicate_expression={"op": "all", "children": [{
        "predicateId": "PRED-RECEIPT", "kind": "auction.contract", "version": 1,
        "hardness": "hard", "arguments": {"contract": "sun"}}]})
    _attach_predicate_evaluation(result, kind="auction.contract")
    value = result["predicateEvaluation"]
    value.update(status="satisfied", satisfied=True)
    value["evidence"][0]["satisfied"] = True
    if mutation == "missing":
        value["evidence"] = []
    elif mutation == "duplicate":
        value["evidence"] *= 2
    elif mutation == "kind":
        value["evidence"][0]["kind"] = "auction.buyer"
    elif mutation == "coercion":
        value["evidence"][0]["satisfied"] = 1
    for row in value["evidence"]:
        _resign_section(row)
    _resign_section(value)
    result["provenance"]["resultIdentity"]["predicateEvaluationHash"] = value["sha256"]
    satisfaction = result["scenarioSatisfaction"]
    satisfaction["predicateEvaluationHash"] = value["sha256"]
    satisfaction["predicates"] = [] if mutation == "projection" else copy.deepcopy(value["evidence"])
    _resign_section(satisfaction)
    result["provenance"]["resultIdentity"]["scenarioSatisfactionHash"] = satisfaction["sha256"]
    _resign_v3_result(result)
    with pytest.raises(RuntimeError, match="predicate|scenario authority"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


@pytest.mark.parametrize("boundary", ["create", "read"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("tactic,mutation", [
    (tactic, mutation)
    for tactic in ("small_card_big_gain", "large_card_sacrifice", "project_bidding", "trump_control")
    for mutation in (None, "profile", "cutoff", "normalization", "snapshot")
] + [(tactic, mutation) for tactic in ("project_bidding", "trump_control")
     for mutation in ("scale", "scaled_threshold")])
def test_v3_small_card_profile_is_bound_to_frozen_snapshot(
    monkeypatch, boundary, wrapped, mutation, tactic,
):
    import sys
    from baloot_platform.tactics import TacticalDetectorProfileV1

    captured = []
    original_builder = build_ortools_research_result_v3

    def capture_inputs(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return original_builder(**kwargs)

    monkeypatch.setattr(sys.modules[__name__], "build_ortools_research_result_v3", capture_inputs)
    _v3_fixture(predicate_expression={"op": "all", "children": [{
        "predicateId": "PRED-RECEIPT", "kind": "tactic.verified", "version": 1,
        "hardness": "report", "arguments": {"kind": tactic,
            "minimumScore": 0, "minimumEvidence": 1}}]}, graph_utility_scale=1000,
        graph_scope="exact" if tactic == "trump_control" else None)
    arguments = captured[0]
    profile = TacticalDetectorProfileV1.defaults()
    small = tactic == "small_card_big_gain"
    project = tactic == "project_bidding"
    trump = tactic == "trump_control"
    field = "projectConsequenceDelta" if project else "minimumCausalGain" if trump else "lowCardMaxStrength" if small else "highCardMinStrength"
    cutoff = profile.project_consequence_delta if project else profile.minimum_causal_gain if trump else profile.low_card_max_strength if small else profile.high_card_min_strength
    changed = 6 if project or trump else 30 if small else 90
    raw = profile.to_dict()
    raw.pop("sha256")
    snapshot = arguments["configuration_snapshot"]
    snapshot["components"]["challenge"]["payload"]["tactical_detectors"] = raw
    _resign_section(snapshot)
    observed = {"detectorId": tactic, "detectorVersion": "3.0.1",
        "detectorProfileHash": profile.sha256, field: cutoff,
        "rootActionCount": 2, **({"projectConsequenceScale": 1000,
            "minimumProjectConsequenceScaled": 5000, "projectConsequenceUnit": "awarded_point_margin"}
            if project else {"causalGainScale": 1000, "minimumCausalGainScaled": 10,
                "causalGainUnit": "awarded_point_margin"} if trump else {"strengthNormalization": "within_suit_0_100"})}
    if mutation == "profile":
        other = TacticalDetectorProfileV1.build({**raw, field: changed})
        observed.update(detectorProfileHash=other.sha256)
        observed[field] = changed
    elif mutation == "cutoff":
        observed[field] = changed
    elif mutation == "normalization":
        observed["projectConsequenceUnit" if project else "causalGainUnit" if trump else "strengthNormalization"] = "whole_deck"
    elif mutation == "snapshot":
        snapshot["components"]["challenge"]["payload"].pop("tactical_detectors")
        _resign_section(snapshot)
    elif mutation == "scale":
        observed["projectConsequenceScale" if project else "causalGainScale"] = 1
    elif mutation == "scaled_threshold":
        observed["minimumProjectConsequenceScaled" if project else "minimumCausalGainScaled"] += 1
    evidence = _signed({"schemaVersion": 1, "predicateId": "PRED-RECEIPT",
        "kind": "tactic.verified", "satisfied": False, "observed": observed})
    if wrapped:
        evidence = _signed({**{key: value for key, value in evidence.items() if key != "sha256"},
            "observed": evidence})
    evaluation = _signed({"schemaVersion": 1, "status": "target_miss", "satisfied": False,
        "evidence": [evidence]})
    if boundary == "create":
        arguments["executor_result"]["predicateEvaluation"] = evaluation
        if mutation is not None:
            with pytest.raises(ValueError, match="native tactical profile"):
                original_builder(**arguments)
            return
        result = original_builder(**arguments)
    else:
        result = original_builder(**arguments)
        _attach_predicate_evaluation(result, kind="tactic.verified")
        result["predicateEvaluation"] = evaluation
        identity = result["provenance"]["resultIdentity"]
        identity["predicateEvaluationHash"] = evaluation["sha256"]
        satisfaction = result["scenarioSatisfaction"]
        satisfaction.update(predicateEvaluationHash=evaluation["sha256"], predicates=[evidence])
        _resign_section(satisfaction)
        identity["scenarioSatisfactionHash"] = satisfaction["sha256"]
        _resign_v3_result(result)
    assert validate_ortools_result_sections_v1(result, recipe=arguments["recipe"]) == result
    kwargs = dict(recipe=arguments["recipe"], documents=arguments["artifact_documents"],
        configuration_snapshot=snapshot)
    if mutation is not None:
        with pytest.raises(ValueError, match="native tactical profile"):
            validate_ortools_result_artifact_documents_v3(result, **kwargs)
    else:
        assert validate_ortools_result_artifact_documents_v3(result, **kwargs) == result


def _native_delayed_profile_readback_fixture(*, tactic="small_card_big_gain", metric="raw_card_points",
        actor_team="north_south", actor_is_target=True, wrapped=False, causal=True):
    from baloot_platform.core.cards import Card
    from baloot_platform.tactics import TacticalDetectorProfileV1

    # Protocol ownership fixture, not an engine-qualified round.
    raw = TacticalDetectorProfileV1.defaults().to_dict()
    raw.pop("sha256")
    raw["nativeDelayedPayoff"] = {"gainMetric": metric, "minimumGain": 2.001}
    profile = TacticalDetectorProfileV1.build(raw)
    actor = 0 if actor_team == "north_south" else 1
    opponent = 1 - actor
    target = actor_team if actor_is_target else "west_east" if actor_team == "north_south" else "north_south"
    actions = ("C8", "C9", "C7") if tactic == "small_card_big_gain" else ("C7", "C8", "CA")
    roots = [{"actionId": Card.parse(card).id, "action": card,
        "value": value if actor_is_target else -value, "satisfies": False}
        for card, value in zip(actions, (-2, -1, 3), strict=True)]
    root_variables = [{"actionId": root["actionId"], "action": root["action"],
        "valueVariable": f"root{index}_value"} for index, root in enumerate(roots)]
    detector = {
        "detectorId": tactic, "detectorVersion": "3.1.0", "detectorProfileHash": profile.sha256,
        "lowCardMaxStrength" if tactic == "small_card_big_gain" else "highCardMinStrength":
            profile.low_card_max_strength if tactic == "small_card_big_gain" else profile.high_card_min_strength,
        "strengthNormalization": "within_suit_0_100", "nativeDelayedPayoff": profile.to_dict()["nativeDelayedPayoff"],
        "nativeGainScale": 1 if metric == "raw_card_points" else 1000,
        "minimumNativeGainScaled": 3 if metric == "raw_card_points" else 2001,
        "rawPayoffUnit": "raw_card_points_excluding_last_trick_bonus", "continuationObjective": "awarded_point_margin",
        "causalProofStatus": "paired_native_delayed_payoff",
        "qualificationSource": "native_cp_sat_paired_delayed_payoff_variables",
        "pairedRawPayoffScope": "canonical_complete_state_continuations", "pairedRawPayoffUnit": "raw_card_points",
        "minimumCausalGain": 0.01, "causalGainScale": 1000,
        "minimumCausalGainScaled": 10, "causalGainUnit": "awarded_point_margin",
        "mechanismKind": tactic, "mechanismScope": "paired_canonical_continuations",
        "comparisonPolicy": "best_actor_alternative_stable_root_order",
        "payoffPointUnit": "raw_card_points_excluding_last_trick_bonus",
        "rootActionCount": 3, "rootValueVariables": [root["valueVariable"] for root in root_variables],
        "selectedRootAction": actions[2], "selectedValueVariable": "root2_value", "bestAlternativeRootIndex": 1,
        "applicable": causal, "verified": causal, "causalProven": causal,
        "entry": causal, "forcedReturn": causal, "partnerEnablement": False,
        "promotion": causal and tactic == "small_card_big_gain", "trumpExhaustion": False,
    }
    pairs = []
    for index, margins in enumerate(((-2, -4), (-3, -5), (-5, 12 if causal else -12))):
        variables = {}
        for trick, margin in zip((7, 8), margins, strict=True):
            for suffix, value in {
                "WinnerSeat": actor if margin > 0 else opponent,
                "LeaderSeat": actor if trick == 7 else opponent,
                "LeadCard": Card.parse(actions[index] if trick == 7 else "C7").id,
                "CardRawMargin": margin, "WinningCard": Card.parse("C7" if margin > 0 else "DA").id,
                "OpponentTrumpsRemaining": 0,
            }.items():
                key = f"trick{trick}{suffix}"
                variables[key] = f"root{index}_{key}"
                detector[f"root{index}{key[0].upper() + key[1:]}"] = value
        pairs.append({"actionId": roots[index]["actionId"], "action": roots[index]["action"],
            "scope": "canonical_complete_state_continuation", "unit": "raw_card_points",
            "firstLeader": actor, "delayedControl": True, "variables": variables})
    for field, suffixes in (("pairedWinnerVariables", ("WinnerSeat",)),
            ("pairedEntryVariables", ("LeaderSeat", "LeadCard", "CardRawMargin")),
            ("pairedDelayedControlVariables", ("WinningCard", "OpponentTrumpsRemaining"))):
        detector[field] = [{key: value for key, value in pair["variables"].items() if key.endswith(suffixes)} for pair in pairs]
    detector.update(nativeGain=(15 if causal else -9) if metric == "raw_card_points" else 4000,
        laterCardMarginGain=17 if causal else -7, firstTrickConcession=2)
    evidence = {"observed": {"observed": detector} if wrapped else detector, "predicateId": "PRED-DELAYED"}
    snapshot = _signed({"bundleHash": _hash("delayed-bundle"),
        "components": {"challenge": {"payload": {"tactical_detectors": raw}}}})
    return {"predicate": {"evidence": [evidence]},
        "proof_outcome": {"rootActions": roots, "selectedAction": actions[2]}, "snapshot": snapshot,
        "bundle_hash": snapshot["bundleHash"], "snapshot_hash": snapshot["sha256"],
        "proof_graph": {"scope": "exact", "utilityScale": 1000},
        "proof_request": {"mode": "exact", "visibility": "complete_state", "utility": "awarded_point_margin", "targetTeam": target},
        "proof_model": {"proofMode": "exact", "visibility": "complete_state", "utility": "awarded_point_margin",
            "rootActorTeam": actor_team, "rootActionVariables": root_variables, "pairedRawPayoffVariables": pairs},
        "backend_version": "3.1.126", "scenario": {"constraints": {"predicateId": "PRED-DELAYED",
            "kind": "tactic.verified", "version": 1, "hardness": "report", "arguments": {"kind": tactic,
                "minimumScore": 0, "minimumEvidence": 1}}, "outcome": {"op": "all", "children": []}}}


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice"])
@pytest.mark.parametrize("actor_team,actor_is_target,wrapped", [
    ("north_south", True, False), ("north_south", False, True),
    ("west_east", True, True), ("west_east", False, False),
])
def test_native_delayed_profile_readback_owns_metrics_pairs_and_actor_orientation(metric, tactic, actor_team, actor_is_target, wrapped):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    _validate_native_tactical_profile_v1(**_native_delayed_profile_readback_fixture(
        metric=metric, tactic=tactic, actor_team=actor_team, actor_is_target=actor_is_target, wrapped=wrapped))


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice"])
def test_native_delayed_profile_readback_preserves_owned_negative_evidence(metric, tactic):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    _validate_native_tactical_profile_v1(**_native_delayed_profile_readback_fixture(metric=metric, tactic=tactic, causal=False))


@pytest.mark.parametrize("opted_in", [False, True])
@pytest.mark.parametrize("roots", [1, 3])
def test_native_delayed_profile_readback_rejects_opted_in_downgrade_except_forced_root(opted_in, roots):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1
    from baloot_platform.tactics import TacticalDetectorProfileV1

    fixture = _native_delayed_profile_readback_fixture()
    detector = fixture["predicate"]["evidence"][0]["observed"]
    detector["detectorVersion"] = "3.0.1"
    if not opted_in:
        raw = fixture["snapshot"]["components"]["challenge"]["payload"]["tactical_detectors"]
        raw.pop("nativeDelayedPayoff")
        detector["detectorProfileHash"] = TacticalDetectorProfileV1.build(raw).sha256
        _resign_section(fixture["snapshot"])
        fixture["snapshot_hash"] = fixture["snapshot"]["sha256"]
    if roots == 1:
        fixture["proof_outcome"]["rootActions"] = fixture["proof_outcome"]["rootActions"][:1]
        fixture["proof_outcome"]["selectedAction"] = fixture["proof_outcome"]["rootActions"][0]["action"]
        detector.update(rootActionCount=1, minimumRequiredRootActions=2,
            inapplicabilityReason="required_alternative_absent", applicable=False, verified=False, causalProven=False)
    if opted_in and roots > 1:
        with pytest.raises(ValueError, match="opted-in delayed detector must use version 3.1.0"):
            _validate_native_tactical_profile_v1(**fixture)
    else:
        _validate_native_tactical_profile_v1(**fixture)


@pytest.mark.parametrize("metric", ["raw_card_points", "awarded_point_margin"])
@pytest.mark.parametrize("section,path,value", [
    ("detector", ("detectorId",), "trump_control"),
    ("detector", ("detectorVersion",), "3.0.0"),
    ("detector", ("detectorProfileHash",), "f" * 64),
    ("detector", ("nativeDelayedPayoff", "gainMetric"), "raw_point_margin"),
    ("detector", ("nativeDelayedPayoff", "minimumGain"), "2.001"),
    ("detector", ("nativeDelayedPayoff", "minimumGain"), 2.002),
    ("detector", ("nativeDelayedPayoff", "enabled"), True),
    ("detector", ("nativeGainScale",), True),
    ("detector", ("minimumNativeGainScaled",), 2002),
    ("detector", ("rawPayoffUnit",), "raw_card_points_with_last_trick_bonus"),
    ("detector", ("continuationObjective",), "raw_card_points"),
    ("detector", ("pairedDelayedControlVariables", 0, "trick7WinningCard"), "unowned_card"),
    ("detector", ("pairedWinnerVariables", 1, "trick8WinnerSeat"), "unowned_winner"),
    ("detector", ("pairedEntryVariables", 2, "trick8CardRawMargin"), "unowned_margin"),
    ("detector", ("bestAlternativeRootIndex",), 0),
    ("detector", ("selectedRootAction",), "C8"),
    ("detector", ("nativeGain",), 14),
    ("detector", ("laterCardMarginGain",), 0),
    ("detector", ("firstTrickConcession",), 0),
    ("detector", ("root2Trick8CardRawMargin",), 13),
    ("detector", ("root1Trick7OpponentTrumpsRemaining",), None),
    ("detector", ("root2Trick8WinningCard",), True),
    ("detector", ("causalProven",), "true"),
    ("proof_request", ("visibility",), "actor_information_set"),
    ("proof_request", ("mode",), "bounded"),
    ("proof_request", ("utility",), "contract_success"),
    ("proof_graph", ("scope",), "bounded"),
    ("proof_model", ("utility",), "raw_card_points"),
    ("proof_model", ("pairedRawPayoffVariables", 0, "delayedControl"), False),
    ("proof_model", ("pairedRawPayoffVariables", 0, "firstLeader"), None),
    ("proof_model", ("rootActionVariables", 0, "action"), "unowned_action"),
    ("proof_outcome", ("rootActions", 1, "value"), 4),
    ("snapshot", ("components", "challenge", "payload", "tactical_detectors", "revision"), "2"),
    ("snapshot", ("components", "challenge", "payload", "tactical_detectors", "nativeDelayedPayoff"), None),
])
def test_native_delayed_profile_readback_rejects_contract_and_solver_tampering(metric, section, path, value):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    fixture = _native_delayed_profile_readback_fixture(metric=metric, tactic="large_card_sacrifice")
    target = fixture["predicate"]["evidence"][0]["observed"] if section == "detector" else fixture[section]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if section == "snapshot":
        _resign_section(fixture["snapshot"])
        fixture["snapshot_hash"] = fixture["snapshot"]["sha256"]
    with pytest.raises(ValueError, match="native tactical profile"):
        _validate_native_tactical_profile_v1(**fixture)


@pytest.mark.parametrize("field", ["nativeDelayedPayoff", "pairedDelayedControlVariables", "pairedWinnerVariables",
    "pairedEntryVariables", "root1Trick7OpponentTrumpsRemaining"])
def test_native_delayed_profile_readback_rejects_missing_owned_evidence(field):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    fixture = _native_delayed_profile_readback_fixture()
    fixture["predicate"]["evidence"][0]["observed"].pop(field)
    with pytest.raises(ValueError, match="native tactical profile"):
        _validate_native_tactical_profile_v1(**fixture)


def test_native_delayed_profile_readback_enforces_stable_awarded_comparator_ties():
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    fixture = _native_delayed_profile_readback_fixture(metric="awarded_point_margin")
    fixture["proof_outcome"]["rootActions"][0]["value"] = -1
    detector = fixture["predicate"]["evidence"][0]["observed"]
    detector.update(bestAlternativeRootIndex=0, laterCardMarginGain=16, firstTrickConcession=3)
    _validate_native_tactical_profile_v1(**fixture)
    detector["bestAlternativeRootIndex"] = 1
    with pytest.raises(ValueError, match="native tactical profile delayed best comparator"):
        _validate_native_tactical_profile_v1(**fixture)


@pytest.mark.parametrize("mutation", ["later_payoff", "large_concession"])
def test_native_delayed_profile_readback_cannot_promote_labels_without_positive_payoff(mutation):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    fixture = _native_delayed_profile_readback_fixture(tactic="large_card_sacrifice", metric="awarded_point_margin",
        causal=mutation != "later_payoff")
    detector = fixture["predicate"]["evidence"][0]["observed"]
    if mutation == "later_payoff":
        detector.update(applicable=True, verified=True, causalProven=True, entry=True)
    else:
        detector.update(root2Trick7CardRawMargin=-3, firstTrickConcession=0)
    with pytest.raises(ValueError, match="native tactical profile delayed causal claim"):
        _validate_native_tactical_profile_v1(**fixture)


@pytest.mark.parametrize("version,roots,accepted", [
    ("2.0.0", 2, True), ("3.0.0", 2, True), ("3.0.1", 1, True), ("3.0.1", 2, False),
])
@pytest.mark.parametrize("tactic", ["small_card_big_gain", "large_card_sacrifice", "project_bidding", "trump_control"])
def test_small_card_profile_legacy_and_owned_forced_root_boundaries(version, roots, accepted, tactic):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    evidence = {"evidence": [{"observed": {"detectorId": tactic,
        "detectorVersion": version, "rootActionCount": 1, "minimumRequiredRootActions": 2,
        "inapplicabilityReason": "required_alternative_absent",
        "applicable": False, "verified": False, "causalProven": False}}]}
    kwargs = dict(bundle_hash="a" * 64, snapshot_hash="b" * 64)
    if accepted:
        _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)
    else:
        with pytest.raises(ValueError, match="snapshot authority"):
            _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)


@pytest.mark.parametrize("boundary", ["create", "read"])
@pytest.mark.parametrize("tactic,mutation", [(tactic, mutation)
    for tactic in ("trump_control", "partner_enablement", "entry_unblocking") for mutation in (None, "version", "detector")]
    + [("partner_enablement", "paired_winner"), ("entry_unblocking", "paired_winner"),
       ("entry_unblocking", "entry_points"), ("entry_unblocking", "entry_variables")])
def test_current_producer_cannot_downgrade_frozen_profile_contract(boundary, tactic, mutation):
    from baloot_platform.tactics import TacticalDetectorProfileV1

    profile = TacticalDetectorProfileV1.defaults().to_dict()
    profile.pop("sha256")
    detector = {"detectorId": tactic, "detectorVersion": "3.0.1",
        "detectorProfileHash": TacticalDetectorProfileV1.defaults().sha256,
        "minimumCausalGain": 0.01, "causalGainScale": 1000,
        "minimumCausalGainScaled": 10, "causalGainUnit": "awarded_point_margin"}
    paired = tactic in {"partner_enablement", "entry_unblocking"}
    roots = [{"actionId": index, "action": card, "value": value, "satisfies": False}
             for index, card, value in ((0, "S7", 0), (1, "S8", -1))]
    model = {"rootActionVariables": roots, "pairedRawPayoffVariables": [
        {"actionId": root["actionId"], "action": root["action"],
         "scope": "canonical_complete_state_continuation", "unit": "raw_card_points",
         "variables": {"trick7WinnerSeat": f"root{index}_winner"}} for index, root in enumerate(roots)]}
    if tactic == "entry_unblocking":
        for index, pair in enumerate(model["pairedRawPayoffVariables"]):
            pair["firstLeader"] = 0
            pair["variables"].update({f"trick7{suffix}": f"root{index}_{suffix}"
                for suffix in ("LeaderSeat", "LeadCard", "CardRawMargin")})
    if paired:
        detector.update(mechanismKind=tactic, mechanismScope="paired_canonical_continuations",
            comparisonPolicy="best_actor_alternative_stable_root_order",
            pairedWinnerVariables=[{key: value for key, value in pair["variables"].items() if key.endswith("WinnerSeat")}
                for pair in model["pairedRawPayoffVariables"]])
    if tactic == "entry_unblocking":
        detector.update(payoffPointUnit="raw_card_points_excluding_last_trick_bonus",
            pairedEntryVariables=[{key: value for key, value in pair["variables"].items() if not key.endswith("WinnerSeat")}
                for pair in model["pairedRawPayoffVariables"]])

    def alter(observed):
        if mutation == "version":
            observed.clear()
            observed.update(detectorId=tactic, detectorVersion="3.0.0")
        elif mutation == "detector":
            observed["detectorId"] = "debate"
        elif mutation == "paired_winner":
            observed["pairedWinnerVariables"] = [{"trick7WinnerSeat": "unowned_winner"}] * 2
        elif mutation == "entry_points":
            observed["payoffPointUnit"] = "raw_points_with_last_trick_bonus"
        elif mutation == "entry_variables":
            observed["pairedEntryVariables"][0]["trick7LeadCard"] = "unowned_lead_card"

    if boundary == "create":
        alter(detector)
    evidence = _signed({"schemaVersion": 1, "predicateId": "PRED-TRUMP", "kind": "tactic.verified",
        "satisfied": False, "evidenceExtractor": "cp_sat.registered_tactic_verified_evidence_v1",
        "observed": detector})
    evaluation = _signed({"schemaVersion": 1, "status": "satisfied", "satisfied": True, "evidence": [evidence]})
    proof = _signed({"completionStatus": "exact_complete", "solverStatus": "optimal", "claimSatisfied": False,
        "nativeQualification": {"PRED-TRUMP": evidence}, **({"rootActions": roots} if paired else {})})

    def create():
        return _v3_fixture(compilation_receipt=True, backend_version="3.1.126" if tactic == "entry_unblocking"
            else "3.1.125" if paired else "3.1.124",
            graph_scope="exact", graph_utility_scale=1000, tactical_profile=profile,
            raw_proof=proof, raw_model=model if paired else None, predicate_evaluation=evaluation,
            predicate_expression={"op": "all", "children": [{"predicateId": "PRED-TRUMP",
                "kind": "tactic.verified", "version": 1, "hardness": "report",
                "arguments": {"kind": tactic, "minimumScore": 0, "minimumEvidence": 1}}]})

    if boundary == "create" and mutation is not None:
        with pytest.raises(ValueError, match="native tactical profile"):
            create()
        return
    result, recipe, documents, campaign = create()
    snapshot = _signed({"schemaVersion": 1, "bundleHash": result["researchAuthority"]["bundleHash"],
        "components": {"challenge": {"payload": {"tactical_detectors": profile}}}})
    assert snapshot["sha256"] == result["provenance"]["resultIdentity"]["configurationSnapshotHash"]
    if boundary == "read" and mutation is not None:
        evaluation = result["predicateEvaluation"]
        row = evaluation["evidence"][0]
        alter(row["observed"])
        _resign_section(row)
        _resign_section(evaluation)
        identity = result["provenance"]["resultIdentity"]
        identity["predicateEvaluationHash"] = evaluation["sha256"]
        satisfaction = result["scenarioSatisfaction"]
        satisfaction.update(predicateEvaluationHash=evaluation["sha256"], predicates=evaluation["evidence"])
        _resign_section(satisfaction)
        identity["scenarioSatisfactionHash"] = satisfaction["sha256"]
        _resign_v3_result(result)
        with pytest.raises(ValueError, match="native tactical profile"):
            validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
                documents=documents, campaign_document=campaign, configuration_snapshot=snapshot)
    else:
        assert result["disposition"] == "rejected"  # Signed protocol fixture, not a qualified round.
        assert validate_ortools_result_artifact_documents_v3(result, recipe=recipe,
            documents=documents, campaign_document=campaign, configuration_snapshot=snapshot) == result


@pytest.mark.parametrize("backend_version,roots,accepted", [
    ("3.1.123", 2, True), ("3.1.124", 1, True), ("3.1.124", 2, False), ("invalid", 2, False),
])
def test_current_profile_exemption_requires_an_owned_forced_root(backend_version, roots, accepted):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    evidence = {"evidence": [{"predicateId": "PRED-TRUMP", "observed": {
        "detectorId": "trump_control", "detectorVersion": "3.0.0", "rootActionCount": 1,
        "minimumRequiredRootActions": 2, "inapplicabilityReason": "required_alternative_absent",
        "applicable": False, "verified": False, "causalProven": False}}]}
    kwargs = {"bundle_hash": "a" * 64, "snapshot_hash": "b" * 64, "backend_version": backend_version,
        "scenario": {"constraints": {"predicateId": "PRED-TRUMP", "kind": "tactic.verified",
            "arguments": {"kind": "trump_control"}}}}
    if accepted:
        _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)
    else:
        with pytest.raises(ValueError, match="detector version"):
            _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)


@pytest.mark.parametrize("tactic,old_version,new_version", [
    ("partner_enablement", "3.1.124", "3.1.125"), ("entry_unblocking", "3.1.125", "3.1.126"),
])
@pytest.mark.parametrize("version,roots,accepted", [
    ("historical", 2, True), ("current", 1, True), ("current", 2, False), ("invalid", 2, False),
])
def test_paired_profile_revision_preserves_historical_and_forced_evidence(tactic, old_version, new_version, version, roots, accepted):
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    evidence = {"evidence": [{"predicateId": "PRED-PARTNER", "observed": {
        "detectorId": tactic, "detectorVersion": "2.0.0", "rootActionCount": 1,
        "minimumRequiredRootActions": 2, "inapplicabilityReason": "required_alternative_absent",
        "applicable": False, "verified": False, "causalProven": False}}]}
    backend_version = old_version if version == "historical" else new_version if version == "current" else version
    kwargs = {"bundle_hash": "a" * 64, "snapshot_hash": "b" * 64, "backend_version": backend_version,
        "scenario": {"constraints": {"predicateId": "PRED-PARTNER", "kind": "tactic.verified",
            "arguments": {"kind": tactic}}}}
    if accepted:
        _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)
    else:
        with pytest.raises(ValueError, match="detector version"):
            _validate_native_tactical_profile_v1(evidence, {"rootActions": [{}] * roots}, None, **kwargs)


def test_current_profile_contract_requires_scenario_ownership():
    from baloot_platform.research.result_authority import _validate_native_tactical_profile_v1

    with pytest.raises(ValueError, match="scenario ownership"):
        _validate_native_tactical_profile_v1(None, {}, None, bundle_hash="a" * 64,
            snapshot_hash="b" * 64, backend_version="3.1.124")


@pytest.mark.parametrize("hardness", ["prefer", "report"])
def test_v3_predicate_readback_does_not_qualify_with_soft_count_children(hardness):
    from baloot_platform.research.result_authority import _validate_predicate_evaluation_v1

    children = [{"predicateId": f"PRED-COUNT-{index}", "kind": "auction.contract",
        "version": 1, "hardness": selected, "arguments": {"contract": "sun"}}
        for index, selected in enumerate(("hard", hardness))]
    scenario = {"constraints": {"op": "count", "children": children, "minimum": 1, "maximum": 1},
        "outcome": {"op": "all", "children": []}}
    value = _signed({"schemaVersion": 1, "status": "satisfied", "satisfied": True,
        "evidence": [_signed({"predicateId": child["predicateId"], "kind": child["kind"],
            "satisfied": True}) for child in children]})
    assert _validate_predicate_evaluation_v1(value, expected_hash=value["sha256"], scenario=scenario) == value
    value["evidence"][0]["satisfied"] = False
    _resign_section(value["evidence"][0])
    _resign_section(value)
    with pytest.raises(ValueError, match="contradicts the hard expression"):
        _validate_predicate_evaluation_v1(value, expected_hash=value["sha256"], scenario=scenario)


def test_v3_read_rejects_coherently_resigned_scenario_substitution() -> None:
    result, recipe = _v3_result()
    substituted = _hash("substituted-scenario")
    identity = result["provenance"]["resultIdentity"]
    identity["scenarioHash"] = substituted
    result["provenance"]["scenarioHash"] = substituted
    satisfaction = result["scenarioSatisfaction"]
    satisfaction["scenarioHash"] = substituted
    _resign_section(satisfaction)
    identity["scenarioSatisfactionHash"] = satisfaction["sha256"]
    _resign_v3_result(result)

    with pytest.raises(RuntimeError, match="recipe identity mismatch"):
        validate_ortools_result_sections_v1(result, recipe=recipe)


def test_v3_read_rejects_satisfied_shape_without_candidate_predicate_and_proof() -> None:
    result, recipe = _v3_result()
    identity = result["provenance"]["resultIdentity"]
    result["terminalStatus"] = "satisfied"
    identity["terminalStatus"] = "satisfied"

    satisfaction = result["scenarioSatisfaction"]
    satisfaction["status"] = "satisfied"
    _resign_section(satisfaction)
    identity["scenarioSatisfactionHash"] = satisfaction["sha256"]

    projection = result["scenarioProjection"]
    projection["status"] = "satisfied"
    _resign_section(projection)
    identity["scenarioProjectionHash"] = projection["sha256"]

    qualification = result["qualification"]
    qualification.update(status="qualified", qualified=True, failures=[])
    _resign_section(qualification)
    identity["qualificationHash"] = qualification["sha256"]
    result["disposition"] = "qualified"
    identity["verdictHash"] = canonical_sha256(
        {
            key: copy.deepcopy(result.get(key))
            for key in (
                "truthLabel",
                "confidence",
                "score",
                "disposition",
                "classification",
                "criticalDecisionIds",
                "interestingSeats",
                "tacticalTags",
                "verifiedTacticalTags",
                "decisionCount",
            )
        }
    )
    _resign_v3_result(result)

    with pytest.raises(RuntimeError, match="qualified shape is incomplete"):
        validate_ortools_result_sections_v1(result, recipe=recipe)

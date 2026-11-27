from __future__ import annotations

import base64
import copy
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from service_test_client import OperatorClient as TestClient

from baloot_platform.research.recipe_api import (
    RecipeListCursorV1,
    ResearchMutationConflictError,
    ResearchMutationReceiptCorruptionError,
    recipe_mutation_intent_hash,
    validate_idempotency_key,
)
from baloot_platform.research.service import ResearchFoundryServiceV1
from baloot_platform.research.store import SQLiteResearchStoreV1


class _ChallengeBundleProxy:
    def __init__(self, bundle: Any, challenge: dict[str, Any]) -> None:
        self._bundle = bundle
        self._challenge = copy.deepcopy(challenge)
        self.sha256 = bundle.sha256
        self.bundle_profile = bundle.bundle_profile
        self.rules = bundle.rules

    def component(self, family: str) -> Any:
        if family == "challenge":
            return copy.deepcopy(self._challenge)
        if family == "rules":
            return copy.deepcopy(self.rules.data)
        return self._bundle.component(family)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bundle, name)


class _FrozenOrToolsContextFactory:
    def __init__(self, authority: Any, context: Any) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.authority = authority
        self.context = context
        self.freeze_calls = 0

    def freeze_selector(
        self, *, bundle_revision_id: str, analysis_mode: str
    ) -> tuple[Any, Any]:
        assert bundle_revision_id == self.context.bundle_revision_id
        assert analysis_mode == "cascaded"
        self.freeze_calls += 1
        return self.authority, self.context

    def resolve_recipe(self, _recipe: Any) -> Any:
        return self.context


def _hash(value: str) -> str:
    return (value.encode("utf-8").hex() + "0" * 64)[:64]


def _recipe(
    code: str,
    *,
    status: str = "published",
    title: str = "وصفة اختبار",
) -> dict[str, Any]:
    return {
        "schemaVersion": 4,
        "code": code,
        "titleAr": title,
        "status": status,
        "sha256": _hash(f"{code}:{status}:{title}"),
    }


def _service(store: SQLiteResearchStoreV1) -> ResearchFoundryServiceV1:
    from baloot_platform.service import api

    base = api.RESEARCH
    revision_id = api.CONTAINER.configuration.resolve(
        bundle_id="baloot_ortools_research_native_v3",
        environment="research-native-v3",
    )["bundle"]["revisionId"]
    authority, context = base.execution_context_factory.freeze_selector(
        bundle_revision_id=str(revision_id),
        analysis_mode="cascaded",
    )
    challenge = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/challenge/intelligent_research.yaml")
        .read_text(encoding="utf-8")
    )
    context = replace(
        context,
        bundle=_ChallengeBundleProxy(context.bundle, challenge),
    )
    return ResearchFoundryServiceV1(
        domain=base.domain,
        compiled_domain=base.compiled_domain,
        store=store,
        artifact_store=base.artifact_store,
        executor=base.executor,
        deal_generator=base.deal_generator,
        auction_compiler=base.auction_compiler,
        escalation_compiler=base.escalation_compiler,
        simulator=base.simulator,
        analyzers=base.analyzers,
        score_profile=base.score_profile,
        decision_score_profile=base.decision_score_profile,
        project_compiler=base.project_compiler,
        play_compiler=base.play_compiler,
        tactical_verifier=base.tactical_verifier,
        maximum_tactical_decisions=base.maximum_tactical_decisions,
        comparison_profile=base.comparison_profile,
        maximum_local_shard_workers=1,
        execution_context_factory=_FrozenOrToolsContextFactory(authority, context),
    )


def _intent(service: ResearchFoundryServiceV1, title: str) -> dict[str, Any]:
    factory = service.execution_context_factory
    assert isinstance(factory, _FrozenOrToolsContextFactory)
    return {
        "schemaVersion": 4,
        "titleAr": title,
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": factory.context.bundle_revision_id,
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "predicateId": "PRED-ROBUST-DEAL",
                "kind": "deal.card_owner",
                "version": 1,
                "hardness": "hard",
                "arguments": {
                    "card": 0,
                    "seat": "north",
                    "handStage": "initial_five",
                },
            },
            "decisionPoint": {"kind": "play", "selector": {}},
            "outcome": {
                "predicateId": "PRED-ROBUST-OUTCOME",
                "kind": "outcome.team_round",
                "version": 1,
                "hardness": "report",
                "arguments": {"team": "north_south", "status": "won"},
            },
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_reachable",
            "targetTeam": "north_south",
            "utility": "awarded_point_margin",
            "successThreshold": -100000,
            "budgetProfile": "late_game_exact_v1",
        },
        "playingQuality": {
            "schemaVersion": 1,
            "mode": "budgeted",
            "decisions": {
                kind: {
                    "utility": "awarded_point_margin",
                    "budgetProfile": (
                        "bounded_v1" if kind == "escalation" else "late_game_exact_v1"
                    ),
                    "horizon": {
                        "unit": "tricks" if kind == "play" else "plies",
                        "value": 1,
                    },
                    "boundaryEvaluator": (
                        "auction_hand_margin_v1"
                        if kind == "bid"
                        else "current_raw_point_margin_v1"
                    ),
                }
                for kind in ("bid", "escalation", "project", "play")
            },
        },
        "resources": {
            "generationLane": "ortools",
            "generationPolicy": "ortools_cp_sat_v3",
            "seedStart": 1,
            "seedCount": 1,
            "seedsPerShard": 1,
            "solverTimeoutSeconds": 1.0,
            "proofBudgetProfile": "late_game_exact_v1",
        },
        "output": {"kind": "laffa", "resultCount": 1},
        "status": "published",
    }


def _decode_cursor(cursor: str) -> dict[str, Any]:
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
    return json.loads(raw.decode("utf-8"))


def _encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def test_recipe_api_key_and_cursor_contracts_fail_closed() -> None:
    assert validate_idempotency_key("recipe-create_1:v1") == "recipe-create_1:v1"
    for invalid in (None, 1, "", " leading", "bad key", "x" * 201):
        with pytest.raises(ValueError, match="Idempotency-Key"):
            validate_idempotency_key(invalid)  # type: ignore[arg-type]

    anchor = RecipeListCursorV1.from_anchor(
        "2026-08-09T01:02:03.123456+00:00",
        "RCP-CURSOR01",
        status="published",
    )
    cursor = anchor.encode()
    assert RecipeListCursorV1.decode(cursor, status="published") == anchor
    with pytest.raises(ValueError, match="status scope"):
        RecipeListCursorV1.decode(cursor, status="retired")

    for malformed in ("", "not!base64", cursor + "=", "e30", None):
        with pytest.raises(ValueError, match="cursor"):
            RecipeListCursorV1.decode(  # type: ignore[arg-type]
                malformed,
                status="published",
            )

    for field, value in (
        ("schemaVersion", 2),
        ("schemaVersion", True),
        ("kind", "other"),
        ("order", "recipe_code_asc"),
        ("createdAt", "2026-08-09T01:02:03+00:00"),
        ("recipeCode", "invalid"),
    ):
        changed = _decode_cursor(cursor)
        changed[field] = value
        with pytest.raises(ValueError, match="cursor"):
            RecipeListCursorV1.decode(
                _encode_cursor(changed),
                status="published",
            )


def test_recipe_create_receipt_is_atomic_restart_safe_and_key_is_not_hashed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "research.sqlite3"
    store = SQLiteResearchStoreV1(database)
    service = _service(store)
    intent = _intent(service, "النية الأصلية")

    first = service.create_recipe(intent, idempotency_key="create-receipt-1")
    factory = service.execution_context_factory
    assert isinstance(factory, _FrozenOrToolsContextFactory)
    assert factory.freeze_calls == 1
    assert first["schemaVersion"] == 4
    assert first["lifecycleStatus"] == "published"
    receipt = store.get_recipe_mutation_receipt("create-receipt-1")
    assert receipt is not None
    assert receipt["mutationKind"] == "create"
    assert receipt["responseStatus"] == 201
    assert receipt["response"] == first
    assert receipt["requestIntentHash"] == recipe_mutation_intent_hash(
        "create", intent
    )
    assert "idempotencyKey" not in store.get_recipe(first["code"])
    original_recipe_hash = first["sha256"]
    store.close()

    restarted = SQLiteResearchStoreV1(database)
    restarted_service = _service(restarted)
    restarted_factory = restarted_service.execution_context_factory
    assert isinstance(restarted_factory, _FrozenOrToolsContextFactory)
    assert restarted_service.create_recipe(
        intent,
        idempotency_key="create-receipt-1",
    ) == first
    assert restarted_factory.freeze_calls == 0
    with pytest.raises(ResearchMutationConflictError, match="different"):
        restarted_service.create_recipe(
            _intent(restarted_service, "نية مختلفة"),
            idempotency_key="create-receipt-1",
        )
    assert restarted_factory.freeze_calls == 0

    second_key = restarted_service.create_recipe(
        intent,
        idempotency_key="create-receipt-2",
    )
    assert second_key == first
    assert second_key["sha256"] == original_recipe_hash
    restarted.close()


def test_recipe_create_and_retire_roll_back_when_receipt_write_fails(
    tmp_path: Path,
) -> None:
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    recipe = _recipe("RCP-ATOMIC01")
    service = _service(store)
    intent = _intent(service, "ذرية")
    store._db.executescript(
        """
        CREATE TRIGGER reject_recipe_receipt
        BEFORE INSERT ON research_recipe_mutation_receipts
        BEGIN
          SELECT RAISE(ABORT, 'receipt write blocked');
        END;
        """
    )
    with pytest.raises(sqlite3.IntegrityError, match="receipt write blocked"):
        service.create_recipe(intent, idempotency_key="atomic-create")
    assert store.list_recipes(limit=10) == []
    assert store.get_recipe_mutation_receipt("atomic-create") is None

    store._db.execute("DROP TRIGGER reject_recipe_receipt")
    store._db.commit()
    store.put_recipe(recipe)
    before = "2026-08-09T00:00:00.000000+00:00"
    store._db.execute(
        "UPDATE research_recipes SET updated_at=? WHERE recipe_code=?",
        (before, recipe["code"]),
    )
    store._db.executescript(
        """
        CREATE TRIGGER reject_retire_receipt
        BEFORE INSERT ON research_recipe_mutation_receipts
        BEGIN
          SELECT RAISE(ABORT, 'retire receipt blocked');
        END;
        """
    )
    with pytest.raises(sqlite3.IntegrityError, match="retire receipt blocked"):
        service.retire_recipe(recipe["code"], idempotency_key="atomic-retire")
    row = store._db.execute(
        "SELECT status,updated_at FROM research_recipes WHERE recipe_code=?",
        (recipe["code"],),
    ).fetchone()
    assert (row["status"], row["updated_at"]) == ("published", before)
    assert store.get_recipe_mutation_receipt("atomic-retire") is None
    store.close()


def test_corrupt_recipe_mutation_receipt_is_an_internal_failure(
    tmp_path: Path,
) -> None:
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    service = _service(store)
    intent = _intent(service, "إيصال تالف")
    service.create_recipe(intent, idempotency_key="corrupt-receipt")
    store._db.execute(
        """UPDATE research_recipe_mutation_receipts
           SET response_json='[]' WHERE idempotency_key=?""",
        ("corrupt-receipt",),
    )
    store._db.commit()
    with pytest.raises(
        ResearchMutationReceiptCorruptionError,
        match="stored research recipe mutation receipt",
    ):
        service.create_recipe(intent, idempotency_key="corrupt-receipt")
    store.close()


def test_recipe_create_receipt_serializes_concurrent_store_instances(
    tmp_path: Path,
) -> None:
    database = tmp_path / "research.sqlite3"
    stores = [SQLiteResearchStoreV1(database), SQLiteResearchStoreV1(database)]
    barrier = threading.Barrier(2)
    services: list[ResearchFoundryServiceV1] = []
    for store in stores:
        services.append(_service(store))
    intent = _intent(services[0], "نفس النية")

    def create(service: ResearchFoundryServiceV1) -> dict[str, Any]:
        barrier.wait(timeout=5)
        return service.create_recipe(
            intent,
            idempotency_key="concurrent-create",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(create, services))
    assert responses[0] == responses[1]
    rows = stores[0]._db.execute(
        "SELECT COUNT(*) AS count FROM research_recipe_mutation_receipts"
    ).fetchone()
    assert rows["count"] == 1
    for store in stores:
        store.close()


def test_recipe_create_receipt_conflicts_concurrent_different_intents(
    tmp_path: Path,
) -> None:
    database = tmp_path / "research.sqlite3"
    stores = [SQLiteResearchStoreV1(database), SQLiteResearchStoreV1(database)]
    barrier = threading.Barrier(2)
    services = [_service(store) for store in stores]
    intents = [
        _intent(services[0], "النية الأولى"),
        _intent(services[1], "النية الثانية"),
    ]

    def create(index: int) -> str:
        barrier.wait(timeout=5)
        try:
            services[index].create_recipe(
                intents[index],
                idempotency_key="concurrent-conflict",
            )
            return "created"
        except ResearchMutationConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(create, range(2)))
    assert sorted(outcomes) == ["conflict", "created"]
    receipt_count = stores[0]._db.execute(
        "SELECT COUNT(*) AS count FROM research_recipe_mutation_receipts"
    ).fetchone()["count"]
    recipe_count = stores[0]._db.execute(
        "SELECT COUNT(*) AS count FROM research_recipes"
    ).fetchone()["count"]
    assert (receipt_count, recipe_count) == (1, 1)
    for store in stores:
        store.close()


def test_recipe_retirement_replays_exactly_without_rewriting_timestamp(
    tmp_path: Path,
) -> None:
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    first_recipe = _recipe("RCP-RETIRE01")
    second_recipe = _recipe("RCP-RETIRE02")
    store.put_recipe(first_recipe)
    store.put_recipe(second_recipe)
    service = _service(store)

    retired = service.retire_recipe(
        first_recipe["code"], idempotency_key="retire-receipt-1"
    )
    first_updated_at = store._db.execute(
        "SELECT updated_at FROM research_recipes WHERE recipe_code=?",
        (first_recipe["code"],),
    ).fetchone()["updated_at"]
    assert retired["lifecycleStatus"] == "retired"
    assert service.retire_recipe(
        first_recipe["code"], idempotency_key="retire-receipt-1"
    ) == retired

    assert service.retire_recipe(
        first_recipe["code"], idempotency_key="retire-receipt-2"
    ) == retired
    second_updated_at = store._db.execute(
        "SELECT updated_at FROM research_recipes WHERE recipe_code=?",
        (first_recipe["code"],),
    ).fetchone()["updated_at"]
    assert second_updated_at == first_updated_at

    with pytest.raises(ResearchMutationConflictError, match="different"):
        service.retire_recipe(
            second_recipe["code"], idempotency_key="retire-receipt-1"
        )
    with pytest.raises(ResearchMutationConflictError, match="different"):
        service.create_recipe(
            {"schemaVersion": 4, "titleAr": "نوع مختلف"},
            idempotency_key="retire-receipt-1",
        )
    receipt = store.get_recipe_mutation_receipt("retire-receipt-1")
    assert receipt is not None and receipt["responseStatus"] == 200
    store.close()


def test_recipe_keyset_pagination_handles_tied_times_two_pages_and_scope(
    tmp_path: Path,
) -> None:
    store = SQLiteResearchStoreV1(tmp_path / "research.sqlite3")
    recipes = [
        _recipe("RCP-PAGEA01"),
        _recipe("RCP-PAGEB01"),
        _recipe("RCP-PAGEC01"),
        _recipe("RCP-PAGED01"),
        _recipe("RCP-PAGER01", status="retired"),
    ]
    for recipe in recipes:
        store.put_recipe(recipe)
    tied = "2026-08-09T02:00:00.000000+00:00"
    older = "2026-08-09T01:00:00.000000+00:00"
    newest = "2026-08-09T03:00:00.000000+00:00"
    for recipe in recipes[:3]:
        store._db.execute(
            "UPDATE research_recipes SET created_at=? WHERE recipe_code=?",
            (tied, recipe["code"]),
        )
    store._db.execute(
        "UPDATE research_recipes SET created_at=? WHERE recipe_code=?",
        (older, recipes[3]["code"]),
    )
    store._db.execute(
        "UPDATE research_recipes SET created_at=? WHERE recipe_code=?",
        (newest, recipes[4]["code"]),
    )
    store._db.commit()
    service = _service(store)

    first = service.list_recipe_page(status="published", limit=2)
    assert [item["code"] for item in first["items"]] == [
        "RCP-PAGEA01",
        "RCP-PAGEB01",
    ]
    assert first["nextCursor"] is not None
    second = service.list_recipe_page(
        status="published",
        cursor=first["nextCursor"],
        limit=1,
    )
    assert [item["code"] for item in second["items"]] == ["RCP-PAGEC01"]
    assert second["nextCursor"] is not None
    final = service.list_recipe_page(
        status="published",
        cursor=second["nextCursor"],
        limit=5,
    )
    assert [item["code"] for item in final["items"]] == ["RCP-PAGED01"]
    assert final["nextCursor"] is None
    assert len(
        {
            item["code"]
            for page in (first, second, final)
            for item in page["items"]
        }
    ) == 4

    with pytest.raises(ValueError, match="status scope"):
        service.list_recipe_page(
            status=None,
            cursor=first["nextCursor"],
            limit=2,
        )
    unfiltered = service.list_recipe_page(limit=5)
    assert unfiltered["items"][0]["code"] == "RCP-PAGER01"
    assert unfiltered["items"][0]["lifecycleStatus"] == "retired"
    assert unfiltered["nextCursor"] is None
    indexes = {
        str(row["name"])
        for row in store._db.execute(
            "PRAGMA index_list('research_recipes')"
        ).fetchall()
    }
    assert {
        "ix_research_recipes_created_page",
        "ix_research_recipes_status_page",
    }.issubset(indexes)
    store.close()


def test_recipe_http_contract_requires_keys_maps_conflict_and_types_cursor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from baloot_platform.service import api

    store = SQLiteResearchStoreV1(tmp_path / "list.sqlite3")
    list_service = _service(store)

    class MutationStub:
        def create_recipe(
            self, _payload: dict[str, Any], *, idempotency_key: str
        ) -> dict[str, Any]:
            assert idempotency_key == "conflict-key"
            raise ResearchMutationConflictError("different intent")

        def retire_recipe(
            self, _code: str, *, idempotency_key: str
        ) -> dict[str, Any]:
            assert idempotency_key == "conflict-key"
            raise ResearchMutationConflictError("different intent")

    body = {
        "schemaVersion": 4,
        "titleAr": "وصفة اختبار",
        "configuration": {
            "schemaVersion": 1,
            "bundleRevisionId": "test-revision",
        },
        "scenario": {
            "schemaVersion": 1,
            "origin": {"kind": "search"},
            "constraints": {
                "predicateId": "PRED-ROBUST-CONTRACT",
                "kind": "auction.contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {"contract": "sun"},
            },
            "decisionPoint": {"kind": "bid", "selector": {"actionIndex": 0}},
            "outcome": {
                "predicateId": "PRED-ROBUST-OUTCOME",
                "kind": "outcome.buyer_contract",
                "version": 1,
                "hardness": "hard",
                "arguments": {"status": "lost"},
            },
        },
        "proof": {
            "schemaVersion": 1,
            "mode": "exact",
            "visibility": "complete_state",
            "claim": "outcome_forced",
            "targetTeam": "north_south",
            "utility": "contract_success",
            "successThreshold": 1,
            "budgetProfile": "late_game_exact_v1",
        },
    }
    monkeypatch.setattr(api, "RESEARCH", MutationStub())
    with TestClient(api.app) as client:
        assert client.post("/api/v2/research-recipes", json=body).status_code == 428
        assert client.post(
            "/api/v2/research-recipes",
            json=body,
            headers={"Idempotency-Key": ""},
        ).status_code == 428
        for invalid in ("bad key", "x" * 201):
            invalid_key = client.post(
                "/api/v2/research-recipes",
                json=body,
                headers={"Idempotency-Key": invalid},
            )
            assert invalid_key.status_code == 422
            assert isinstance(invalid_key.json()["detail"], str)
        invalid_body = client.post(
            "/api/v2/research-recipes",
            json={},
            headers={"Idempotency-Key": "body-validation"},
        )
        assert invalid_body.status_code == 422
        assert isinstance(invalid_body.json()["detail"], list)
        assert client.post(
            "/api/v2/research-recipes",
            json=body,
            headers={"Idempotency-Key": "conflict-key"},
        ).status_code == 409
        assert client.post(
            "/api/v2/research-recipes/RCP-CONFLICT:retire"
        ).status_code == 428
        assert client.post(
            "/api/v2/research-recipes/RCP-CONFLICT:retire",
            headers={"Idempotency-Key": "conflict-key"},
        ).status_code == 409

    monkeypatch.setattr(api, "RESEARCH", list_service)
    with TestClient(api.app) as client:
        terminal = client.get("/api/v2/research-recipes?limit=1")
        assert terminal.status_code == 200
        assert terminal.json() == {"items": [], "nextCursor": None}
        malformed = client.get("/api/v2/research-recipes?cursor=e30")
        assert malformed.status_code == 422
    store.close()

    schema = api.app.openapi()
    for path in (
        "/api/v2/research-recipes",
        "/api/v2/research-recipes/{code}:retire",
    ):
        operation = schema["paths"][path]["post"]
        header = next(
            item
            for item in operation["parameters"]
            if item["in"] == "header" and item["name"] == "Idempotency-Key"
        )
        assert header["required"] is True
        assert header["schema"]["maxLength"] == 200
        assert {"409", "422", "428"}.issubset(operation["responses"])
    assert "404" in schema["paths"][
        "/api/v2/research-recipes/{code}:retire"
    ]["post"]["responses"]
    list_operation = schema["paths"]["/api/v2/research-recipes"]["get"]
    cursor_parameter = next(
        item for item in list_operation["parameters"] if item["name"] == "cursor"
    )
    cursor_string = next(
        item
        for item in cursor_parameter["schema"]["anyOf"]
        if item.get("type") == "string"
    )
    assert cursor_string["maxLength"] == 500
    page_schema = schema["components"]["schemas"]["ResearchRecipeListResponse"]
    assert page_schema["additionalProperties"] is False
    assert set(page_schema["required"]) == {"items", "nextCursor"}
    assert {item["type"] for item in page_schema["properties"]["nextCursor"]["anyOf"]} == {
        "string",
        "null",
    }
    validation_schema = schema["components"]["schemas"][
        "ResearchMutationValidationErrorResponse"
    ]
    assert len(validation_schema["anyOf"]) == 2


def test_research_job_mutations_publish_exact_idempotency_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from baloot_platform.service import api

    class JobMutationStub:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        def submit(self, recipe_code: str, *, idempotency_key: str):
            self.calls.append(("run", recipe_code, idempotency_key))
            if recipe_code == "RCP-MISSING":
                raise KeyError(recipe_code)
            raise ValueError("run intent conflicts")

        def create_comparison_campaign(
            self, recipe_code: str, *, idempotency_key: str
        ):
            self.calls.append(("comparison", recipe_code, idempotency_key))
            if recipe_code == "RCP-MISSING":
                raise KeyError(recipe_code)
            raise ValueError("campaign intent conflicts")

    stub = JobMutationStub()
    monkeypatch.setattr(api, "RESEARCH", stub)
    with TestClient(api.app) as client:
        for path in ("/api/v2/research-runs", "/api/v2/comparison-campaigns"):
            assert client.post(
                path, json={"recipeCode": "RCP-MISSING"}
            ).status_code == 428
            assert client.post(
                path,
                json={"recipeCode": "RCP-MISSING"},
                headers={"Idempotency-Key": "   "},
            ).status_code == 428
            invalid_body = client.post(
                path,
                json={},
                headers={"Idempotency-Key": "body-validation"},
            )
            assert invalid_body.status_code == 422
            assert isinstance(invalid_body.json()["detail"], list)

        assert client.post(
            "/api/v2/research-runs",
            json={"recipeCode": "RCP-MISSING"},
            headers={"Idempotency-Key": "  run-key  "},
        ).status_code == 404
        assert client.post(
            "/api/v2/research-runs",
            json={"recipeCode": "RCP-CONFLICT"},
            headers={"Idempotency-Key": "run-conflict"},
        ).status_code == 409
        assert client.post(
            "/api/v2/comparison-campaigns",
            json={"recipeCode": "RCP-MISSING"},
            headers={"Idempotency-Key": "  comparison-key  "},
        ).status_code == 404
        assert client.post(
            "/api/v2/comparison-campaigns",
            json={"recipeCode": "RCP-CONFLICT"},
            headers={"Idempotency-Key": "comparison-conflict"},
        ).status_code == 409

    assert stub.calls == [
        ("run", "RCP-MISSING", "run-key"),
        ("run", "RCP-CONFLICT", "run-conflict"),
        ("comparison", "RCP-MISSING", "comparison-key"),
        ("comparison", "RCP-CONFLICT", "comparison-conflict"),
    ]

    schema = api.app.openapi()
    for path in ("/api/v2/research-runs", "/api/v2/comparison-campaigns"):
        operation = schema["paths"][path]["post"]
        header = next(
            item
            for item in operation["parameters"]
            if item["in"] == "header" and item["name"] == "Idempotency-Key"
        )
        assert header["required"] is True
        assert header["schema"] == {
            "type": "string",
            "minLength": 1,
            "pattern": r".*\S.*",
        }
        assert {"202", "404", "409", "422", "428"}.issubset(
            operation["responses"]
        )
        for status in ("404", "428"):
            assert operation["responses"][status]["content"][
                "application/json"
            ]["schema"] == {
                "$ref": "#/components/schemas/ResearchMutationErrorResponse"
            }
        expected_conflict_schema = (
            "ResearchRunConflictResponse"
            if path == "/api/v2/research-runs"
            else "ResearchMutationErrorResponse"
        )
        assert operation["responses"]["409"]["content"]["application/json"][
            "schema"
        ] == {"$ref": f"#/components/schemas/{expected_conflict_schema}"}
        assert operation["responses"]["422"]["content"][
            "application/json"
        ]["schema"] == {
            "$ref": "#/components/schemas/ResearchRequestValidationErrorResponse"
        }

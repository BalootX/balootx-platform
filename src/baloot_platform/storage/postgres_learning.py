from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from baloot_platform.learning.challenge_publication import (
    validate_evidence_sha256,
    validate_governed_transition,
    validate_manual_status_transition,
    validate_reviewer_role,
)
from baloot_platform.history.provenance_v4 import game_serial
from baloot_platform.storage.codec_v7 import decode as decode_payload, encode as encode_payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value) if isinstance(value, str) else value


def _iso(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class PostgresLearningStore:
    """PostgreSQL parity for immutable learning/history metadata.

    API nodes and workers can share this store safely.  Large model, dataset,
    replay and video bytes remain in the configured artifact store; trusted
    session snapshots are bytea because reopening requires the exact engine
    object graph.  All public/queryable metadata is canonical JSONB.
    """

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=16, open=True)

    @staticmethod
    def _public_game(session: Any) -> dict[str, Any]:
        return session.to_dict(reveal_hands=False)

    @staticmethod
    def _trusted_game(session: Any) -> dict[str, Any]:
        return session.to_dict(reveal_hands=True)

    def _one(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def _all(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())

    def archive_game(self, session: Any, *, bundle_hash: str, engine_version: str, parent_game_hash: str | None = None, status: str = "completed", label: str | None = None) -> dict[str, Any]:
        public = self._public_game(session)
        trusted = self._trusted_game(session)
        state_hash = _hash(trusted)
        identity = {"gameId": session.id, "version": int(session.version), "stateHash": state_hash, "bundleHash": bundle_hash, "engineVersion": engine_version, "parentGameHash": parent_game_hash}
        digest = _hash(identity)
        serial = game_serial(digest)
        payload = encode_payload(session)
        created = _now()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO archived_games(game_hash,serial,game_id,parent_game_hash,bundle_hash,engine_version,state_hash,public_json,trusted_payload,status,label,created_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::timestamptz)
                   ON CONFLICT(game_hash) DO NOTHING""",
                (digest, serial, session.id, parent_game_hash, bundle_hash, engine_version, state_hash, _canonical(public), payload, status, label, created),
            )
            conn.execute("UPDATE learning_decisions SET game_hash=%s WHERE game_id=%s AND game_hash IS NULL", (digest, session.id))
        return {"gameHash": digest, "serial": serial, "gameId": session.id, "parentGameHash": parent_game_hash, "bundleHash": bundle_hash, "engineVersion": engine_version, "stateHash": state_hash, "status": status, "createdAt": created}

    def _game_row(self, identifier: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM archived_games WHERE game_hash=%s OR serial=%s OR game_id=%s ORDER BY created_at DESC LIMIT 1", (identifier, identifier, identifier))

    def get_game(self, identifier: str) -> dict[str, Any] | None:
        row = self._game_row(identifier)
        if row is None:
            return None
        state = _json(row["public_json"], {})
        return {
            "gameHash": row["game_hash"], "serial": row["serial"], "gameId": row["game_id"],
            "parentGameHash": row["parent_game_hash"], "bundleHash": row["bundle_hash"],
            "engineVersion": row["engine_version"], "stateHash": row["state_hash"], "status": row["status"],
            "label": row["label"], "createdAt": _iso(row["created_at"]), "state": state,
            "decisions": self.list_decisions(str(row["game_hash"])),
            "artifacts": self.list_artifacts("game", str(row["game_hash"])),
            "audits": self.list_game_audits(str(row["game_hash"])),
            "videoRecords": self.list_video_records("game", str(row["game_hash"])),
        }

    def list_games(self, *, query: str = "", offset: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        rows = self._all(
            """SELECT * FROM archived_games WHERE %s='' OR game_hash ILIKE %s OR serial ILIKE %s OR game_id ILIKE %s OR COALESCE(label,'') ILIKE %s
               ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (
                query, pattern, pattern, pattern, pattern,
                max(1, min(5000, int(limit))), max(0, int(offset)),
            ),
        )
        result = []
        for row in rows:
            state = _json(row["public_json"], {})
            result.append({
                "gameHash": row["game_hash"], "serial": row["serial"], "gameId": row["game_id"],
                "parentGameHash": row["parent_game_hash"], "bundleHash": row["bundle_hash"],
                "engineVersion": row["engine_version"], "stateHash": row["state_hash"], "status": row["status"],
                "label": row["label"], "createdAt": _iso(row["created_at"]), "contract": state.get("contract"),
                "buyer": state.get("buyer"), "roundNumber": state.get("roundNumber"), "matchScore": state.get("matchScore"),
            })
        return result

    def reopen_session(self, identifier: str) -> Any:
        row = self._game_row(identifier)
        if row is None:
            raise KeyError(identifier)
        return decode_payload(bytes(row["trusted_payload"]))

    def verify_game(self, identifier: str) -> dict[str, Any]:
        row = self._game_row(identifier)
        if row is None:
            return {"valid": False, "error": "game not found"}
        try:
            session = decode_payload(bytes(row["trusted_payload"]))
            state_hash = _hash(self._trusted_game(session))
            digest = _hash({"gameId": session.id, "version": int(session.version), "stateHash": state_hash, "bundleHash": row["bundle_hash"], "engineVersion": row["engine_version"], "parentGameHash": row["parent_game_hash"]})
            return {"valid": state_hash == row["state_hash"] and digest == row["game_hash"], "gameHash": row["game_hash"], "serial": row["serial"], "computedGameHash": digest, "computedStateHash": state_hash}
        except Exception as exc:
            return {"valid": False, "error": str(exc), "gameHash": row["game_hash"]}

    def record_decision(
        self, *, game_id: str, game_version: int, mode: str, state: Any, actor: Any,
        decision_kind: str, legal_actions: Iterable[Any], selected_action: Any,
        bidding: Any | None = None, decision_metadata: dict[str, Any] | None = None,
        actor_source: str | None = None, actor_profile: str | None = None,
        strategy_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from baloot_platform.analysis.models import InformationState
        from baloot_platform.core.cards import Card
        actor_name = actor.name.lower() if hasattr(actor, "name") else str(actor)
        metadata = copy.deepcopy(decision_metadata or strategy_trace or {})
        if actor_source is not None:
            metadata.setdefault("source", str(actor_source))
        if actor_profile is not None:
            metadata.setdefault("profileId", str(actor_profile))
        metadata.setdefault("source", "human")
        metadata.setdefault("observerSafe", True)
        metadata.setdefault("schemaVersion", 2)
        if decision_kind == "play":
            public_state = InformationState.from_complete(state, actor).to_dict()
            legal = [Card.parse(item).to_dict() for item in legal_actions]
            selected = Card.parse(selected_action).to_dict()
        else:
            public_state = {"rulesHash": state.rules_hash, "observer": actor_name, "observerHand": [card.to_dict() for card in state.hand(actor)], "purchaseCard": state.purchase_card.to_dict() if state.purchase_card else None, "dealer": state.dealer.name.lower(), "matchScore": list(state.match_score), "biddingHistory": [item.to_dict() for item in state.bid_history], "biddingRound": getattr(bidding, "round_index", None), "biddingActor": actor_name}
            legal = [copy.deepcopy(item) for item in legal_actions]
            selected = copy.deepcopy(selected_action)
        ordinal_row = self._one("SELECT COALESCE(MAX(ordinal),0)+1 AS value FROM learning_decisions WHERE game_id=%s", (game_id,))
        ordinal = int(ordinal_row["value"])
        identity = {"gameId": game_id, "gameVersion": int(game_version), "ordinal": ordinal, "kind": decision_kind, "actor": actor_name, "publicState": public_state, "legalActions": legal, "selectedAction": selected, "decisionMetadata": metadata}
        digest = _hash(identity)
        decision_id = f"DEC-{digest[:20].upper()}"
        trusted = encode_payload({"mode": mode, "state": state.clone(), "actor": actor, "decisionKind": decision_kind, "bidding": copy.deepcopy(bidding), "legalActions": tuple(copy.deepcopy(tuple(legal_actions))), "selectedAction": copy.deepcopy(selected_action), "decisionMetadata": metadata})
        created = _now()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """INSERT INTO learning_decisions(decision_id,game_id,game_hash,game_version,ordinal,decision_kind,actor,phase,legal_actions_json,selected_action_json,public_state_json,trusted_payload,decision_hash,decision_metadata_json,created_at)
                   VALUES(%s,%s,NULL,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb,%s::timestamptz) ON CONFLICT(decision_id) DO NOTHING""",
                (decision_id, game_id, int(game_version), ordinal, decision_kind, actor_name, state.phase.value, _canonical(legal), _canonical(selected), _canonical(public_state), trusted, digest, _canonical(metadata), created),
            )
        return {"decisionId": decision_id, "gameId": game_id, "gameVersion": int(game_version), "ordinal": ordinal, "decisionKind": decision_kind, "actor": actor_name, "phase": state.phase.value, "legalActions": legal, "selectedAction": selected, "decisionHash": digest, "decisionMetadata": copy.deepcopy(metadata), "createdAt": created}

    def _decision_row(self, decision_id: str) -> dict[str, Any] | None:
        return self._one("SELECT * FROM learning_decisions WHERE decision_id=%s", (decision_id,))

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        row = self._decision_row(decision_id)
        if row is None:
            return None
        return {"decisionId": row["decision_id"], "gameId": row["game_id"], "gameHash": row["game_hash"], "gameVersion": row["game_version"], "ordinal": row["ordinal"], "decisionKind": row["decision_kind"], "actor": row["actor"], "phase": row["phase"], "legalActions": _json(row["legal_actions_json"], []), "selectedAction": _json(row["selected_action_json"]), "publicState": _json(row["public_state_json"], {}), "decisionHash": row["decision_hash"], "decisionMetadata": _json(row.get("decision_metadata_json"), {}), "createdAt": _iso(row["created_at"]), "analyses": self.list_analyses(decision_id), "feedback": self.list_feedback(decision_id), "structuredReviews": self.list_structured_reviews(decision_id), "reviewConsensus": self.review_consensus(decision_id), "artifacts": self.list_artifacts("decision", decision_id), "audits": self.list_decision_audits(decision_id), "difficulty": self.get_difficulty_item(decision_id), "videoRecords": self.list_video_records("decision", decision_id)}

    def load_decision_payload(self, decision_id: str) -> dict[str, Any]:
        row = self._decision_row(decision_id)
        if row is None:
            raise KeyError(decision_id)
        return decode_payload(bytes(row["trusted_payload"]))

    def list_decisions(self, identifier: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._all("SELECT decision_id FROM learning_decisions WHERE game_id=%s OR game_hash=%s ORDER BY ordinal LIMIT %s", (identifier, identifier, max(1, min(10000, int(limit)))))
        return [item for row in rows if (item := self.get_decision(str(row["decision_id"]))) is not None]

    def decision_ledger(self, identifier: str):
        from baloot_platform.challenge.evidence_v3 import DecisionLedgerV1, canonical_sha256

        game = self.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        decisions = self.list_decisions(identifier, limit=10000)
        if not decisions:
            raise ValueError("game has no recorded decisions")
        enriched: list[dict[str, Any]] = []
        for decision in decisions:
            trusted = self.load_decision_payload(str(decision["decisionId"]))
            state = trusted.get("state")
            if state is None or not hasattr(state, "to_dict"):
                raise ValueError("decision trusted state is unavailable")
            enriched.append({
                **decision,
                "completeStateHash": canonical_sha256(state.to_dict(reveal_hands=True)),
            })
        return DecisionLedgerV1.build(
            str(game.get("gameHash") or game.get("gameId") or identifier), enriched
        )

    def complete_game_trace(self, identifier: str):
        from baloot_platform.challenge.trace import CompleteGameTraceV1

        game = self.get_game(identifier)
        if game is None:
            raise KeyError(identifier)
        verification = self.verify_game(identifier)
        if not verification.get("valid"):
            raise ValueError("archived game failed integrity verification")
        session = self.reopen_session(identifier)
        decisions = self.list_decisions(identifier, limit=10000)
        models = {
            str((item.get("decisionMetadata") or {}).get("modelId"))
            for item in decisions
            if (item.get("decisionMetadata") or {}).get("modelId")
        }
        return CompleteGameTraceV1.build(
            game_record=game,
            trusted_final_state=session.to_dict(reveal_hands=True),
            decision_ledger=self.decision_ledger(identifier),
            model_releases=models,
        )

    def add_analysis(self, decision_id: str, *, engine_version: str, bundle_hash: str, config: dict[str, Any], result: dict[str, Any], supersedes_analysis_id: str | None = None) -> dict[str, Any]:
        if self._decision_row(decision_id) is None:
            raise KeyError(decision_id)
        config_hash, result_hash = _hash(config), _hash(result)
        analysis_id = f"ANL-{_hash({'decisionId': decision_id, 'engineVersion': engine_version, 'bundleHash': bundle_hash, 'configHash': config_hash, 'resultHash': result_hash})[:20].upper()}"
        created = _now()
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_analyses(analysis_id,decision_id,engine_version,bundle_hash,config_hash,result_hash,config_json,result_json,supersedes_analysis_id,created_at)
                            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::timestamptz) ON CONFLICT(analysis_id) DO NOTHING""", (analysis_id, decision_id, engine_version, bundle_hash, config_hash, result_hash, _canonical(config), _canonical(result), supersedes_analysis_id, created))
        return {"analysisId": analysis_id, "decisionId": decision_id, "engineVersion": engine_version, "bundleHash": bundle_hash, "configHash": config_hash, "resultHash": result_hash, "config": copy.deepcopy(config), "result": copy.deepcopy(result), "supersedesAnalysisId": supersedes_analysis_id, "createdAt": created}

    def list_analyses(self, decision_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM learning_analyses WHERE decision_id=%s ORDER BY created_at DESC,analysis_id DESC", (decision_id,))
        return [{"analysisId": row["analysis_id"], "decisionId": row["decision_id"], "engineVersion": row["engine_version"], "bundleHash": row["bundle_hash"], "configHash": row["config_hash"], "resultHash": row["result_hash"], "config": _json(row["config_json"], {}), "result": _json(row["result_json"], {}), "supersedesAnalysisId": row["supersedes_analysis_id"], "createdAt": _iso(row["created_at"])} for row in rows]

    def add_feedback(self, decision_id: str, *, preferred_action: Any, engine_action: Any | None, confidence: float, rationale: str, source: str, reviewer: str | None = None, interestingness: float = 0.0, model_error: float = 0.0) -> dict[str, Any]:
        if self._decision_row(decision_id) is None:
            raise KeyError(decision_id)
        confidence = max(0.0, min(1.0, float(confidence)))
        feedback_id = f"FDB-{_hash({'decisionId': decision_id, 'preferredAction': preferred_action, 'source': source, 'reviewer': reviewer, 'rationale': rationale, 'nonce': uuid.uuid4().hex})[:20].upper()}"
        created = _now()
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_feedback(feedback_id,decision_id,preferred_action_json,engine_action_json,confidence,rationale,source,reviewer,interestingness,model_error,created_at)
                            VALUES(%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::timestamptz)""", (feedback_id, decision_id, _canonical(preferred_action), _canonical(engine_action) if engine_action is not None else None, confidence, rationale.strip(), source, reviewer, max(0.0,min(1.0,float(interestingness))), max(0.0,min(1.0,float(model_error))), created))
        return {"feedbackId": feedback_id, "decisionId": decision_id, "preferredAction": copy.deepcopy(preferred_action), "engineAction": copy.deepcopy(engine_action), "confidence": confidence, "rationale": rationale.strip(), "source": source, "reviewer": reviewer, "interestingness": max(0.0,min(1.0,float(interestingness))), "modelError": max(0.0,min(1.0,float(model_error))), "createdAt": created}

    def list_feedback(self, decision_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM learning_feedback WHERE decision_id=%s ORDER BY created_at DESC,feedback_id DESC", (decision_id,))
        return [{"feedbackId": row["feedback_id"], "decisionId": row["decision_id"], "preferredAction": _json(row["preferred_action_json"]), "engineAction": _json(row["engine_action_json"]), "confidence": row["confidence"], "rationale": row["rationale"], "source": row["source"], "reviewer": row["reviewer"], "interestingness": row["interestingness"], "modelError": row["model_error"], "createdAt": _iso(row["created_at"])} for row in rows]

    def learning_queue(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._all("""SELECT d.decision_id,COUNT(f.feedback_id) AS feedback_count,COALESCE(MAX(f.interestingness),0) AS interestingness,COALESCE(MAX(f.model_error),0) AS model_error,COALESCE(AVG(f.confidence),0) AS confidence
                            FROM learning_decisions d LEFT JOIN learning_feedback f ON f.decision_id=d.decision_id GROUP BY d.decision_id
                            ORDER BY (0.35*COALESCE(MAX(f.interestingness),0)+0.30*COALESCE(MAX(f.model_error),0)+0.20*(1-COALESCE(AVG(f.confidence),0))+0.15/(1+COUNT(f.feedback_id))) DESC LIMIT %s""", (max(1,min(5000,int(limit))),))
        result = []
        for row in rows:
            decision = self.get_decision(str(row["decision_id"]))
            if decision is None:
                continue
            base_score = 0.30*float(row["interestingness"])+0.25*float(row["model_error"])+0.15*(1-float(row["confidence"]))+0.10/(1+int(row["feedback_count"]))
            audit = (decision.get("audits") or [None])[0] or {}
            regret = float(audit.get("expectedRegret", 0.0)); win_drop = float(audit.get("winProbabilityDrop", 0.0))
            severity = max(1.0 if audit.get("catastrophicCrossing") else 0.0, min(1.0, regret/25.0), min(1.0, win_drop/0.35))
            audit_uncertainty = 1.0-max(0.0,min(1.0,float(audit.get("auditConfidence",0.0)))) if audit else 0.5
            difficulty = decision.get("difficulty") or {}; difficulty_under_reviewed = 1.0/(1.0+int(difficulty.get("responseCount",0)))
            score = base_score+0.15*severity+0.03*audit_uncertainty+0.02*difficulty_under_reviewed
            result.append({"decision":decision,"feedbackCount":int(row["feedback_count"]),"auditSeverity":severity,"auditUncertainty":audit_uncertainty,"difficultyResponseCount":int(difficulty.get("responseCount",0)),"activeLearningScore":max(0.0,min(1.0,score))})
        return result

    def register_dataset(self, artifact: Any, *, purpose: str, manifest: dict[str, Any], artifact_path: str | None = None) -> dict[str, Any]:
        created = _now()
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_datasets(dataset_id,purpose,artifact_path,manifest_json,example_count,sha256,created_at)
                            VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s::timestamptz) ON CONFLICT(dataset_id) DO UPDATE SET manifest_json=excluded.manifest_json""", (artifact.id,purpose,(artifact_path or str(artifact.examples_path)),_canonical(manifest),int(artifact.example_count),artifact.sha256,created))
        return {**artifact.to_dict(), "examples": str(artifact.examples_path), "artifactPath": artifact_path or str(artifact.examples_path), "purpose": purpose, "createdAt": created, "manifestData": manifest}

    def list_datasets(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM learning_datasets ORDER BY created_at DESC")
        return [{"id": row["dataset_id"], "purpose": row["purpose"], "artifactPath": row["artifact_path"], "manifest": _json(row["manifest_json"], {}), "exampleCount": row["example_count"], "sha256": row["sha256"], "createdAt": _iso(row["created_at"])} for row in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list_datasets() if item["id"] == dataset_id), None)

    def register_model(self, *, model_id: str, plugin_id: str, version: str, artifact_path: str, manifest: dict[str, Any], dataset_id: str | None, status: str = "candidate", metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _now()
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_models(model_id,plugin_id,version,artifact_path,manifest_json,metrics_json,dataset_id,status,created_at,updated_at)
                            VALUES(%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::timestamptz,%s::timestamptz)
                            ON CONFLICT(model_id) DO UPDATE SET plugin_id=excluded.plugin_id,version=excluded.version,artifact_path=excluded.artifact_path,manifest_json=excluded.manifest_json,metrics_json=excluded.metrics_json,dataset_id=excluded.dataset_id,status=excluded.status,updated_at=excluded.updated_at""", (model_id,plugin_id,version,artifact_path,_canonical(manifest),_canonical(metrics) if metrics is not None else None,dataset_id,status,now,now))
        return self.get_model(model_id) or {}

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM learning_models WHERE model_id=%s", (model_id,))
        if row is None:
            return None
        return {"modelId": row["model_id"], "pluginId": row["plugin_id"], "version": row["version"], "artifactPath": row["artifact_path"], "manifest": _json(row["manifest_json"], {}), "metrics": _json(row["metrics_json"], {}), "datasetId": row["dataset_id"], "status": row["status"], "createdAt": _iso(row["created_at"]), "updatedAt": _iso(row["updated_at"])}

    def list_models(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT model_id FROM learning_models ORDER BY updated_at DESC")
        return [item for row in rows if (item := self.get_model(str(row["model_id"]))) is not None]

    def update_model(self, model_id: str, *, status: str | None = None, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.get_model(model_id) is None:
            raise KeyError(model_id)
        with self.pool.connection() as conn, conn.transaction():
            if status == "production":
                conn.execute("UPDATE learning_models SET status='validated',updated_at=now() WHERE status='production' AND model_id<>%s", (model_id,))
            conn.execute("""UPDATE learning_models SET status=COALESCE(%s,status),metrics_json=CASE WHEN %s IS NULL THEN metrics_json ELSE %s::jsonb END,updated_at=now() WHERE model_id=%s""", (status, _canonical(metrics) if metrics is not None else None, _canonical(metrics) if metrics is not None else None, model_id))
        return self.get_model(model_id) or {}

    def add_structured_review(self, review: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(review)
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO learning_structured_reviews(
                     review_id,decision_id,reviewer_id,reviewer_role,preferred_action_json,
                     confidence,reason_codes_json,rationale,evidence_json,view_mode,source,
                     engine_validation,engine_regret,engine_win_drop,review_hash,created_at
                   ) VALUES(%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::timestamptz)
                   ON CONFLICT(review_id) DO NOTHING""",
                (value["reviewId"], value["decisionId"], value["reviewerId"], value["reviewerRole"],
                 _canonical(value.get("preferredAction")), float(value.get("confidence", 0.0)),
                 _canonical(value.get("reasonCodes") or []), str(value.get("rationale") or ""),
                 _canonical(value.get("evidence") or []), str(value.get("viewMode") or "observer"),
                 str(value.get("source") or "decision_workbench"), str(value.get("engineValidation") or "unverified"),
                 value.get("engineRegret"), value.get("engineWinDrop"), value["reviewHash"], value["createdAt"]),
            )
        return next((item for item in self.list_structured_reviews(value["decisionId"]) if item["reviewId"] == value["reviewId"]), value)

    def list_structured_reviews(self, decision_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM learning_structured_reviews WHERE decision_id=%s ORDER BY created_at DESC,review_id DESC", (decision_id,))
        return [{
            "reviewId": row["review_id"], "decisionId": row["decision_id"],
            "reviewerId": row["reviewer_id"], "reviewerRole": row["reviewer_role"],
            "preferredAction": _json(row["preferred_action_json"]), "confidence": float(row["confidence"]),
            "reasonCodes": _json(row["reason_codes_json"], []), "rationale": row["rationale"],
            "evidence": _json(row["evidence_json"], []), "viewMode": row["view_mode"],
            "source": row["source"], "engineValidation": row["engine_validation"],
            "engineRegret": row["engine_regret"], "engineWinDrop": row["engine_win_drop"],
            "reviewHash": row["review_hash"], "createdAt": _iso(row["created_at"]),
        } for row in rows]

    def review_consensus(self, decision_id: str) -> dict[str, Any]:
        from baloot_platform.audit.models import action_key
        from baloot_platform.review.models import ReviewRole
        reviews = self.list_structured_reviews(decision_id)
        buckets: dict[str, dict[str, Any]] = {}; total = 0.0
        for item in reviews:
            try: role_weight = ReviewRole(str(item.get("reviewerRole", "player"))).weight
            except Exception: role_weight = 1.0
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            validation_weight = {"supported": 1.0, "plausible": .9, "uncertain": .65, "unverified": .5, "refuted_by_current_engine": .25}.get(str(item.get("engineValidation")), .5)
            weight = role_weight * confidence * validation_weight; key = action_key(item.get("preferredAction"))
            bucket = buckets.setdefault(key, {"action": item.get("preferredAction"), "weight": 0.0, "reviewers": set(), "roles": set(), "reasons": {}})
            bucket["weight"] += weight; bucket["reviewers"].add(str(item.get("reviewerId"))); bucket["roles"].add(str(item.get("reviewerRole")))
            for reason in item.get("reasonCodes") or []: bucket["reasons"][str(reason)] = bucket["reasons"].get(str(reason), 0) + 1
            total += weight
        ordered = sorted(buckets.values(), key=lambda value: value["weight"], reverse=True)
        if not ordered: return {"decisionId": decision_id, "status": "no_reviews", "trainingEligible": False, "reviewCount": 0, "independentReviewers": 0}
        best = ordered[0]; share = best["weight"] / total if total else 0.0; independent = len(best["reviewers"]); expert = bool(best["roles"] & {"expert", "referee", "coach"}); eligible = share >= .65 and independent >= 2 and (expert or independent >= 3)
        return {"decisionId": decision_id, "status": "consensus" if eligible else "collecting", "preferredAction": best["action"], "supportShare": round(share, 6), "reviewCount": len(reviews), "independentReviewers": independent, "expertOrRefereeIncluded": expert, "trainingEligible": eligible, "reasonCounts": dict(sorted(best["reasons"].items())), "feedbackIsEvidenceNotTruth": True}

    def put_challenge(self, *, challenge_id: str, decision_id: str | None, fingerprint: str, archetype: str, publishability: float, payload: dict[str, Any], episode: Any | None, status: str = "candidate") -> dict[str, Any]:
        now = _now(); episode_payload = encode_payload(episode) if episode is not None else None
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_challenges(challenge_id,decision_id,fingerprint,archetype,publishability,status,payload_json,episode_payload,created_at,updated_at)
                            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::timestamptz,%s::timestamptz)
                            ON CONFLICT(fingerprint) DO UPDATE SET publishability=GREATEST(learning_challenges.publishability,excluded.publishability),payload_json=CASE WHEN excluded.publishability>=learning_challenges.publishability THEN excluded.payload_json ELSE learning_challenges.payload_json END,episode_payload=CASE WHEN excluded.publishability>=learning_challenges.publishability THEN excluded.episode_payload ELSE learning_challenges.episode_payload END,updated_at=excluded.updated_at""", (challenge_id,decision_id,fingerprint,archetype,max(0.0,min(1.0,float(publishability))),status,_canonical(payload),episode_payload,now,now))
        row = self._one("SELECT challenge_id FROM learning_challenges WHERE fingerprint=%s", (fingerprint,))
        return self.get_challenge(str(row["challenge_id"])) if row else {}

    def get_challenge(self, challenge_id: str) -> dict[str, Any] | None:
        row = self._one("SELECT * FROM learning_challenges WHERE challenge_id=%s", (challenge_id,))
        if row is None:
            return None
        return {"challengeId": row["challenge_id"], "decisionId": row["decision_id"], "fingerprint": row["fingerprint"], "archetype": row["archetype"], "publishability": row["publishability"], "status": row["status"], "payload": _json(row["payload_json"], {}), "createdAt": _iso(row["created_at"]), "updatedAt": _iso(row["updated_at"]), "artifacts": self.list_artifacts("challenge", challenge_id)}

    def load_challenge_episode(self, challenge_id: str) -> Any:
        row = self._one("SELECT episode_payload FROM learning_challenges WHERE challenge_id=%s", (challenge_id,))
        if row is None or row["episode_payload"] is None:
            raise KeyError(challenge_id)
        return decode_payload(bytes(row["episode_payload"]))

    def list_challenges(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if status:
            rows = self._all("SELECT challenge_id FROM learning_challenges WHERE status=%s ORDER BY publishability DESC,created_at DESC LIMIT %s", (status,max(1,min(1000,int(limit)))))
        else:
            rows = self._all("SELECT challenge_id FROM learning_challenges ORDER BY publishability DESC,created_at DESC LIMIT %s", (max(1,min(1000,int(limit))),))
        return [item for row in rows if (item := self.get_challenge(str(row["challenge_id"]))) is not None]

    def update_challenge_status(self, challenge_id: str, status: str) -> dict[str, Any]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT status FROM learning_challenges "
                "WHERE challenge_id=%s FOR UPDATE",
                (challenge_id,),
            ).fetchone()
            if row is None:
                raise KeyError(challenge_id)
            validate_manual_status_transition(str(row[0]), status)
            result = conn.execute("UPDATE learning_challenges SET status=%s,updated_at=now() WHERE challenge_id=%s", (status,challenge_id))
            if result.rowcount != 1:
                raise RuntimeError("challenge status update lost its target")
        return self.get_challenge(challenge_id) or {}

    def record_challenge_review(self, challenge_id: str, *, reviewer_id: str, reviewer_role: str,
                                decision: str, evidence_sha256: str, comment: str,
                                policy: dict[str, Any]) -> dict[str, Any]:
        role=validate_reviewer_role(reviewer_role,policy); verdict=str(decision).strip().lower()
        if not str(reviewer_id).strip(): raise ValueError("challenge reviewer identity is required")
        if verdict not in {"approve","reject"}: raise ValueError("challenge review decision must be approve or reject")
        evidence=validate_evidence_sha256(evidence_sha256); note=str(comment).strip()
        if len(note)>2000: raise ValueError("challenge review comment exceeds 2000 characters")
        identity={"challengeId":challenge_id,"reviewerId":reviewer_id,"decision":verdict,"evidenceSha256":evidence}
        review_id=f"CRV-{_hash(identity)[:20].upper()}"; created=_now()
        with self.pool.connection() as conn, conn.transaction():
            row=conn.execute("SELECT status FROM learning_challenges WHERE challenge_id=%s FOR UPDATE",(challenge_id,)).fetchone()
            if row is None: raise KeyError(challenge_id)
            if str(row[0])!="review": raise ValueError("challenge reviews are accepted only while status is review")
            try:
                conn.execute("""INSERT INTO learning_challenge_publication_reviews_v21
                    (review_id,challenge_id,reviewer_id,reviewer_role,decision,evidence_sha256,comment,created_at)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::timestamptz)""",
                    (review_id,challenge_id,reviewer_id,role,verdict,evidence,note,created))
            except Exception as exc:
                if getattr(exc,"sqlstate",None)=="23505": raise ValueError("reviewer has already submitted an immutable review") from exc
                raise
        return {"reviewId":review_id,**identity,"reviewerRole":role,"comment":note,"createdAt":created}

    def list_challenge_reviews(self, challenge_id: str) -> list[dict[str, Any]]:
        rows=self._all("SELECT * FROM learning_challenge_publication_reviews_v21 WHERE challenge_id=%s ORDER BY created_at,review_id",(challenge_id,))
        return [{"reviewId":r["review_id"],"challengeId":r["challenge_id"],"reviewerId":r["reviewer_id"],"reviewerRole":r["reviewer_role"],"decision":r["decision"],"evidenceSha256":r["evidence_sha256"],"comment":r["comment"],"createdAt":_iso(r["created_at"])} for r in rows]

    def govern_challenge_status(self, challenge_id: str, *, target: str, actor_id: str,
                                actor_role: str, evidence_sha256: str,
                                policy: dict[str, Any]) -> dict[str, Any]:
        evidence=validate_evidence_sha256(evidence_sha256)
        with self.pool.connection() as conn, conn.transaction():
            row=conn.execute("SELECT * FROM learning_challenges WHERE challenge_id=%s FOR UPDATE",(challenge_id,)).fetchone()
            if row is None: raise KeyError(challenge_id)
            challenge={"status":row[5],"publishability":row[4],"archetype":row[3],"payload":_json(row[6],{}),"actorId":actor_id}
            reviews=self.list_challenge_reviews(challenge_id)
            validate_governed_transition(challenge,reviews,target=target,actor_role=actor_role,policy=policy)
            created=_now(); identity={"challengeId":challenge_id,"actorId":actor_id,"fromStatus":challenge["status"],"toStatus":target,"evidenceSha256":evidence}; event_id=f"CPE-{_hash(identity)[:20].upper()}"
            conn.execute("UPDATE learning_challenges SET status=%s,updated_at=%s::timestamptz WHERE challenge_id=%s",(target,created,challenge_id))
            conn.execute("""INSERT INTO learning_challenge_publication_events_v21
                (event_id,challenge_id,actor_id,actor_role,from_status,to_status,evidence_sha256,created_at)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s::timestamptz)""",(event_id,challenge_id,actor_id,actor_role,challenge["status"],target,evidence,created))
        return self.get_challenge(challenge_id) or {}

    def list_challenge_publication_events(self, challenge_id: str) -> list[dict[str, Any]]:
        rows=self._all("""SELECT * FROM learning_challenge_publication_events_v21 WHERE challenge_id=%s
            ORDER BY CASE to_status WHEN 'approved' THEN 1 WHEN 'published' THEN 2 ELSE 3 END,created_at,event_id""",(challenge_id,))
        return [{"eventId":r["event_id"],"challengeId":r["challenge_id"],"actorId":r["actor_id"],"actorRole":r["actor_role"],"fromStatus":r["from_status"],"toStatus":r["to_status"],"evidenceSha256":r["evidence_sha256"],"createdAt":_iso(r["created_at"])} for r in rows]

    def link_artifact(self, *, owner_kind: str, owner_id: str, kind: str, uri: str, sha256: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        identity = {"ownerKind": owner_kind,"ownerId": owner_id,"kind": kind,"uri": uri,"sha256": sha256}
        artifact_id = f"ART-{_hash(identity)[:20].upper()}"; created = _now()
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_artifacts(artifact_id,owner_kind,owner_id,kind,uri,sha256,metadata_json,created_at)
                            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::timestamptz) ON CONFLICT(artifact_id) DO UPDATE SET metadata_json=excluded.metadata_json""", (artifact_id,owner_kind,owner_id,kind,uri,sha256,_canonical(metadata or {}),created))
        return {"artifactId": artifact_id, **identity, "metadata": metadata or {}, "createdAt": created}

    def list_artifacts(self, owner_kind: str, owner_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM learning_artifacts WHERE owner_kind=%s AND owner_id=%s ORDER BY created_at DESC", (owner_kind,owner_id))
        return [{"artifactId": row["artifact_id"], "ownerKind": row["owner_kind"], "ownerId": row["owner_id"], "kind": row["kind"], "uri": row["uri"], "sha256": row["sha256"], "metadata": _json(row["metadata_json"], {}), "createdAt": _iso(row["created_at"])} for row in rows]

    # Replay audit, calibration, and timecode parity with LearningStore.
    def put_decision_audit(self, report: dict[str, Any]) -> dict[str, Any]:
        decision_id=str(report["decisionId"]); audit_hash=str(report.get("auditHash") or _hash(report)); audit_id=str(report.get("decisionAuditId") or f"DAU-{audit_hash[:20].upper()}"); created=str(report.get("createdAt") or _now()); payload={**copy.deepcopy(report),"decisionAuditId":audit_id,"auditHash":audit_hash,"createdAt":created}
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_decision_audits(decision_audit_id,decision_id,audit_hash,quality_label,quality_score,report_json,created_at) VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s::timestamptz) ON CONFLICT(audit_hash) DO NOTHING""",(audit_id,decision_id,audit_hash,str(payload.get("qualityLabel","unknown")),float(payload.get("qualityScore",0)),_canonical(payload),created))
        return payload

    save_decision_audit=put_decision_audit

    def get_decision_audit(self, identifier: str) -> dict[str, Any] | None:
        row=self._one("SELECT report_json FROM learning_decision_audits WHERE decision_audit_id=%s OR audit_hash=%s ORDER BY created_at DESC LIMIT 1",(identifier,identifier))
        return _json(row["report_json"],{}) if row else None

    def list_decision_audits(self, decision_id: str, *, limit: int=100) -> list[dict[str, Any]]:
        return [_json(row["report_json"],{}) for row in self._all("SELECT report_json FROM learning_decision_audits WHERE decision_id=%s ORDER BY created_at DESC LIMIT %s",(decision_id,max(1,min(5000,int(limit)))))]

    def put_game_audit(self, report: dict[str, Any]) -> dict[str, Any]:
        report_hash=str(report.get("reportHash") or _hash(report)); audit_id=str(report.get("auditId") or f"AUD-{report_hash[:20].upper()}"); created=str(report.get("createdAt") or _now()); payload={**copy.deepcopy(report),"auditId":audit_id,"reportHash":report_hash,"createdAt":created}
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_game_audits(audit_id,game_id,game_hash,report_hash,mode,bundle_hash,engine_version,report_json,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::timestamptz) ON CONFLICT(report_hash) DO NOTHING""",(audit_id,str(payload.get("gameId","")),payload.get("gameHash"),report_hash,str(payload.get("mode","observer_safe")),str(payload.get("bundleHash","")),str(payload.get("engineVersion","")),_canonical(payload),created))
        return payload

    save_game_audit=put_game_audit

    def get_game_audit(self, identifier: str) -> dict[str, Any] | None:
        row=self._one("SELECT report_json FROM learning_game_audits WHERE audit_id=%s OR report_hash=%s ORDER BY created_at DESC LIMIT 1",(identifier,identifier))
        return _json(row["report_json"],{}) if row else None

    def list_game_audits(self, identifier: str, *, limit: int=100) -> list[dict[str, Any]]:
        row=self._game_row(identifier); game_id=str(row["game_id"]) if row else identifier; game_hash=str(row["game_hash"]) if row else identifier
        return [_json(item["report_json"],{}) for item in self._all("SELECT report_json FROM learning_game_audits WHERE game_id=%s OR game_hash=%s ORDER BY created_at DESC LIMIT %s",(game_id,game_hash,max(1,min(5000,int(limit)))))]

    def put_difficulty_item(self, item: dict[str, Any]) -> dict[str, Any]:
        calibrated=item.get("calibratedDifficulty",item.get("calibratedScore",item.get("engineDifficulty",50))); updated=str(item.get("updatedAt") or _now())
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_difficulty_items(challenge_id,engine_difficulty,latent_difficulty,discrimination,response_count,distinct_players,standard_error,calibrated_score,label,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::timestamptz) ON CONFLICT(challenge_id) DO UPDATE SET engine_difficulty=excluded.engine_difficulty,latent_difficulty=excluded.latent_difficulty,discrimination=excluded.discrimination,response_count=excluded.response_count,distinct_players=excluded.distinct_players,standard_error=excluded.standard_error,calibrated_score=excluded.calibrated_score,label=excluded.label,updated_at=excluded.updated_at""",(str(item["challengeId"]),float(item.get("engineDifficulty",calibrated)),float(item.get("latentDifficulty",0)),float(item.get("discrimination",1)),int(item.get("responseCount",0)),int(item.get("distinctPlayers",0)),float(item.get("standardError",1)),float(calibrated),str(item.get("label","medium")),updated))
        return self.get_difficulty_item(str(item["challengeId"])) or copy.deepcopy(item)

    upsert_difficulty_item=put_difficulty_item

    def get_difficulty_item(self, challenge_id: str) -> dict[str, Any] | None:
        row=self._one("SELECT * FROM learning_difficulty_items WHERE challenge_id=%s",(challenge_id,))
        if not row:return None
        return {"challengeId":row["challenge_id"],"engineDifficulty":row["engine_difficulty"],"latentDifficulty":row["latent_difficulty"],"discrimination":row["discrimination"],"responseCount":row["response_count"],"distinctPlayers":row["distinct_players"],"standardError":row["standard_error"],"calibratedDifficulty":row["calibrated_score"],"calibratedScore":row["calibrated_score"],"label":row["label"],"updatedAt":_iso(row["updated_at"])}

    def list_difficulty_items(self, *, limit: int=1000) -> list[dict[str, Any]]:
        return [item for row in self._all("SELECT challenge_id FROM learning_difficulty_items ORDER BY updated_at DESC LIMIT %s",(max(1,min(10000,int(limit))),)) if (item:=self.get_difficulty_item(str(row["challenge_id"]))) is not None]

    def put_difficulty_player(self, player: dict[str, Any]) -> dict[str, Any]:
        updated=str(player.get("updatedAt") or _now())
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_difficulty_players(player_id,ability,response_count,standard_error,updated_at) VALUES(%s,%s,%s,%s,%s::timestamptz) ON CONFLICT(player_id) DO UPDATE SET ability=excluded.ability,response_count=excluded.response_count,standard_error=excluded.standard_error,updated_at=excluded.updated_at""",(str(player["playerId"]),float(player.get("ability",0)),int(player.get("responseCount",0)),float(player.get("standardError",1)),updated))
        return self.get_difficulty_player(str(player["playerId"])) or copy.deepcopy(player)

    upsert_difficulty_player=put_difficulty_player

    def get_difficulty_player(self, player_id: str) -> dict[str, Any] | None:
        row=self._one("SELECT * FROM learning_difficulty_players WHERE player_id=%s",(player_id,))
        return None if not row else {"playerId":row["player_id"],"ability":row["ability"],"responseCount":row["response_count"],"standardError":row["standard_error"],"updatedAt":_iso(row["updated_at"])}

    def add_difficulty_response(self, response: dict[str, Any]) -> dict[str, Any]:
        response_id=str(response.get("responseId") or f"DRS-{_hash({**response,'nonce':uuid.uuid4().hex})[:20].upper()}"); created=str(response.get("createdAt") or _now()); payload={**copy.deepcopy(response),"responseId":response_id,"createdAt":created}
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_difficulty_responses(response_id,challenge_id,player_id,correct,response_seconds,confidence,chosen_action_json,trace_json,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::timestamptz)""",(response_id,str(response["challengeId"]),str(response["playerId"]),bool(response.get("correct")),response.get("responseSeconds"),response.get("confidence"),_canonical(response.get("chosenAction")) if response.get("chosenAction") is not None else None,_canonical(response.get("trace") or {}),created))
        return payload

    def list_difficulty_responses(self, challenge_id: str | None=None, *, player_id: str | None=None, limit: int=1000) -> list[dict[str, Any]]:
        where=[]; values=[]
        if challenge_id:where.append("challenge_id=%s");values.append(challenge_id)
        if player_id:where.append("player_id=%s");values.append(player_id)
        query="SELECT * FROM learning_difficulty_responses"+(" WHERE "+" AND ".join(where) if where else "")+" ORDER BY created_at DESC LIMIT %s";values.append(max(1,min(10000,int(limit))))
        rows=self._all(query,tuple(values))
        return [{"responseId":r["response_id"],"challengeId":r["challenge_id"],"playerId":r["player_id"],"correct":bool(r["correct"]),"responseSeconds":r["response_seconds"],"confidence":r["confidence"],"chosenAction":_json(r["chosen_action_json"]),"trace":_json(r["trace_json"],{}),"createdAt":_iso(r["created_at"])} for r in rows]

    def put_video_record(self, record: dict[str, Any]) -> dict[str, Any]:
        identity={"ownerKind":record.get("ownerKind"),"ownerId":record.get("ownerId"),"episodeHash":record.get("episodeHash"),"rendererId":record.get("rendererId"),"timelineHash":_hash(record.get("timeline") or {})};video_id=str(record.get("videoId") or f"VID-{_hash(identity)[:20].upper()}");created=str(record.get("createdAt") or _now());payload={**copy.deepcopy(record),"videoId":video_id,"createdAt":created}
        with self.pool.connection() as conn:
            conn.execute("""INSERT INTO learning_video_records(video_id,owner_kind,owner_id,game_hash,episode_hash,renderer_id,manifest_json,timeline_json,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::timestamptz) ON CONFLICT(video_id) DO UPDATE SET manifest_json=excluded.manifest_json,timeline_json=excluded.timeline_json""",(video_id,str(record.get("ownerKind","episode")),str(record.get("ownerId","")),record.get("gameHash"),str(record.get("episodeHash","")),str(record.get("rendererId","")),_canonical(record.get("manifest") or {}),_canonical(record.get("timeline") or {}),created))
        return payload

    def get_video_record(self, video_id: str) -> dict[str, Any] | None:
        row=self._one("SELECT * FROM learning_video_records WHERE video_id=%s",(video_id,))
        return None if not row else {"videoId":row["video_id"],"ownerKind":row["owner_kind"],"ownerId":row["owner_id"],"gameHash":row["game_hash"],"episodeHash":row["episode_hash"],"rendererId":row["renderer_id"],"manifest":_json(row["manifest_json"],{}),"timeline":_json(row["timeline_json"],{}),"createdAt":_iso(row["created_at"])}

    def list_video_records(self, owner_kind: str | None=None, owner_id: str | None=None, *, game_hash: str | None=None, limit: int=100) -> list[dict[str, Any]]:
        where=[];values=[]
        if owner_kind:where.append("owner_kind=%s");values.append(owner_kind)
        if owner_id:where.append("owner_id=%s");values.append(owner_id)
        if game_hash:where.append("game_hash=%s");values.append(game_hash)
        query="SELECT video_id FROM learning_video_records"+(" WHERE "+" AND ".join(where) if where else "")+" ORDER BY created_at DESC LIMIT %s";values.append(max(1,min(5000,int(limit))))
        return [item for row in self._all(query,tuple(values)) if (item:=self.get_video_record(str(row["video_id"]))) is not None]

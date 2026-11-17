"""Opaque, observer-safe locator for exact duplicate challenges."""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import re
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from baloot_platform.core.enums import Seat
from baloot_platform.core.rules import RuleBook

from .perspective_pipeline import build_observer_projection
from .publication import ChallengeAnswerEvidenceV1
from .replay import TraceDecisionProjectionStoreV1
from .trace import CompleteGameTraceV1


_ACTOR_PROMPT = "choose_acting_action"
_OBSERVER_PROMPTS = {
    "predict_opponent_play",
    "read_partner_signal",
    "identify_threat",
    "plan_next_response",
    "infer_distribution",
    "post_reveal_analysis",
}

_PLAYER_NAME_SEATS = frozenset(seat.name.lower() for seat in Seat)
_RESEARCH_RESULT_CODE = re.compile(r"LAF-[A-F0-9]{10}")
_RESEARCH_EXACT_STATE_ID = re.compile(r"STATE-[A-F0-9]{20}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SEAT_ORDER = ("north", "west", "south", "east")
_RESEARCH_CHALLENGE_COPY_KEYS = frozenset(
    {
        "action_accessibility",
        "action_bid_ashkal",
        "action_bid_hokum",
        "action_bid_pass",
        "action_bid_sun",
        "action_escalation_decline",
        "action_escalation_double",
        "action_escalation_four",
        "action_escalation_gahwa",
        "action_escalation_triple",
        "action_mode_locked",
        "action_mode_open",
        "action_project_declare",
        "action_project_fifty",
        "action_project_four_hundred",
        "action_project_hundred_four_kind",
        "action_project_hundred_sequence",
        "action_project_siri",
        "decision_options_aria",
        "proof_boundary_bounded",
        "proof_boundary_exact",
        "proof_boundary_policy_conditioned",
        "prompt_bid",
        "prompt_escalation",
        "prompt_project",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ChallengeLocatorV1:
    code: str
    universal_link: str
    observer_seat: str
    acting_seat: str
    prompt_contract: str
    reveal_policy: str
    expires_at: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "code": self.code,
            "universalLink": self.universal_link,
            "observerSeat": self.observer_seat,
            "actingSeat": self.acting_seat,
            "promptContract": self.prompt_contract,
            "revealPolicy": self.reveal_policy,
            "expiresAt": self.expires_at,
        }


class ChallengeAttemptConflictError(RuntimeError):
    """A participant already committed a different answer for this item."""


class SQLiteChallengeLocatorStore:
    """Local control-plane store; private bindings never enter public responses."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS challenge_locators_v1 (
                    code TEXT PRIMARY KEY,
                    binding_json TEXT NOT NULL,
                    public_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    revoked_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS challenge_locator_attempts_v1 (
                    attempt_id TEXT PRIMARY KEY,
                    code TEXT NOT NULL,
                    participant_key TEXT NOT NULL,
                    chosen_action_json TEXT NOT NULL,
                    response_seconds REAL,
                    confidence REAL,
                    changed_answer INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    UNIQUE(code, participant_key),
                    FOREIGN KEY(code) REFERENCES challenge_locators_v1(code)
                );
                CREATE INDEX IF NOT EXISTS challenge_locator_attempts_code_v1_idx
                    ON challenge_locator_attempts_v1(code, created_at, attempt_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def put(
        self,
        code: str,
        *,
        binding: Mapping[str, Any],
        public: Mapping[str, Any],
        created_at: int,
        expires_at: int | None,
    ) -> None:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT binding_json,public_json,expires_at FROM challenge_locators_v1 WHERE code=?",
                (code,),
            ).fetchone()
            values = (_canonical(binding), _canonical(public), expires_at)
            if existing is not None:
                if (existing["binding_json"], existing["public_json"], existing["expires_at"]) != values:
                    raise RuntimeError("opaque challenge code collision")
                return
            conn.execute(
                """INSERT INTO challenge_locators_v1(
                       code,binding_json,public_json,created_at,expires_at
                   ) VALUES(?,?,?,?,?)""",
                (code, values[0], values[1], int(created_at), expires_at),
            )

    def get(self, code: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM challenge_locators_v1 WHERE code=?",
                (str(code).upper(),),
            ).fetchone()
        if row is None:
            return None
        return {
            "code": row["code"],
            "binding": json.loads(row["binding_json"]),
            "public": json.loads(row["public_json"]),
            "createdAt": int(row["created_at"]),
            "expiresAt": int(row["expires_at"]) if row["expires_at"] is not None else None,
            "revokedAt": int(row["revoked_at"]) if row["revoked_at"] is not None else None,
        }

    def revoke(self, code: str, *, revoked_at: int) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """UPDATE challenge_locators_v1 SET revoked_at=?
                   WHERE code=? AND revoked_at IS NULL""",
                (int(revoked_at), str(code).upper()),
            )
        return result.rowcount == 1

    def commit_attempt(
        self,
        *,
        attempt_id: str,
        code: str,
        participant_key: str,
        chosen_action: Mapping[str, Any],
        response_seconds: float | None,
        confidence: float | None,
        changed_answer: bool,
        created_at: int,
    ) -> dict[str, Any]:
        action_json = _canonical(chosen_action)
        with self._lock, self._connect() as conn:
            inserted = conn.execute(
                """INSERT OR IGNORE INTO challenge_locator_attempts_v1(
                       attempt_id,code,participant_key,chosen_action_json,
                       response_seconds,confidence,changed_answer,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    attempt_id,
                    str(code).upper(),
                    participant_key,
                    action_json,
                    response_seconds,
                    confidence,
                    int(bool(changed_answer)),
                    int(created_at),
                ),
            )
            row = conn.execute(
                """SELECT * FROM challenge_locator_attempts_v1
                   WHERE code=? AND participant_key=?""",
                (str(code).upper(), participant_key),
            ).fetchone()
        if row is None:  # pragma: no cover - defensive database boundary
            raise RuntimeError("challenge attempt commit was not persisted")
        if row["chosen_action_json"] != action_json:
            raise ChallengeAttemptConflictError(
                "participant already committed a different answer"
            )
        return {**self._attempt_row(row), "newlyCommitted": inserted.rowcount == 1}

    @staticmethod
    def _attempt_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "attemptId": row["attempt_id"],
            "code": row["code"],
            "participantKey": row["participant_key"],
            "chosenAction": json.loads(row["chosen_action_json"]),
            "responseSeconds": row["response_seconds"],
            "confidence": row["confidence"],
            "changedAnswer": bool(row["changed_answer"]),
            "createdAt": int(row["created_at"]),
        }

    def list_attempts(self, code: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM challenge_locator_attempts_v1
                   WHERE code=? ORDER BY created_at,attempt_id""",
                (str(code).upper(),),
            ).fetchall()
        return [self._attempt_row(row) for row in rows]

    def close(self) -> None:
        return None


class PostgresChallengeLocatorStore:
    """Multi-node locator store backed by the governed PostgreSQL schema."""

    def __init__(self, dsn: str):
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('Install with: pip install "baloot-platform[postgres]"') from exc
        self.pool = ConnectionPool(dsn, min_size=1, max_size=8, open=True)

    def close(self) -> None:
        self.pool.close()

    def put(
        self,
        code: str,
        *,
        binding: Mapping[str, Any],
        public: Mapping[str, Any],
        created_at: int,
        expires_at: int | None,
    ) -> None:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                "SELECT binding_json,public_json,expires_at FROM challenge_locators_v1 WHERE code=%s FOR UPDATE",
                (code,),
            )
            existing = cursor.fetchone()
            values = (_canonical(binding), _canonical(public), expires_at)
            if existing is not None:
                stored_binding = existing["binding_json"]
                stored_public = existing["public_json"]
                if not isinstance(stored_binding, str):
                    stored_binding = _canonical(stored_binding)
                if not isinstance(stored_public, str):
                    stored_public = _canonical(stored_public)
                if (stored_binding, stored_public, existing["expires_at"]) != values:
                    raise RuntimeError("opaque challenge code collision")
                return
            cursor.execute(
                """INSERT INTO challenge_locators_v1(
                       code,binding_json,public_json,created_at,expires_at
                   ) VALUES(%s,%s::jsonb,%s::jsonb,%s,%s)""",
                (code, values[0], values[1], int(created_at), expires_at),
            )

    def get(self, code: str) -> dict[str, Any] | None:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM challenge_locators_v1 WHERE code=%s", (str(code).upper(),))
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "code": row["code"],
            "binding": json.loads(row["binding_json"]) if isinstance(row["binding_json"], str) else row["binding_json"],
            "public": json.loads(row["public_json"]) if isinstance(row["public_json"], str) else row["public_json"],
            "createdAt": int(row["created_at"]),
            "expiresAt": int(row["expires_at"]) if row["expires_at"] is not None else None,
            "revokedAt": int(row["revoked_at"]) if row["revoked_at"] is not None else None,
        }

    def revoke(self, code: str, *, revoked_at: int) -> bool:
        with self.pool.connection() as conn:
            result = conn.execute(
                """UPDATE challenge_locators_v1 SET revoked_at=%s
                   WHERE code=%s AND revoked_at IS NULL""",
                (int(revoked_at), str(code).upper()),
            )
        return result.rowcount == 1

    @staticmethod
    def _attempt_row(row: Mapping[str, Any]) -> dict[str, Any]:
        action = row["chosen_action_json"]
        return {
            "attemptId": row["attempt_id"],
            "code": row["code"],
            "participantKey": row["participant_key"],
            "chosenAction": json.loads(action) if isinstance(action, str) else action,
            "responseSeconds": row["response_seconds"],
            "confidence": row["confidence"],
            "changedAnswer": bool(row["changed_answer"]),
            "createdAt": int(row["created_at"]),
        }

    def commit_attempt(
        self,
        *,
        attempt_id: str,
        code: str,
        participant_key: str,
        chosen_action: Mapping[str, Any],
        response_seconds: float | None,
        confidence: float | None,
        changed_answer: bool,
        created_at: int,
    ) -> dict[str, Any]:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn, conn.transaction(), conn.cursor(row_factory=dict_row) as cursor:
            action_json = _canonical(chosen_action)
            cursor.execute(
                """INSERT INTO challenge_locator_attempts_v1(
                       attempt_id,code,participant_key,chosen_action_json,
                       response_seconds,confidence,changed_answer,created_at
                   ) VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                   ON CONFLICT(code,participant_key) DO NOTHING""",
                (
                    attempt_id,
                    str(code).upper(),
                    participant_key,
                    action_json,
                    response_seconds,
                    confidence,
                    bool(changed_answer),
                    int(created_at),
                ),
            )
            newly_committed = cursor.rowcount == 1
            cursor.execute(
                """SELECT * FROM challenge_locator_attempts_v1
                   WHERE code=%s AND participant_key=%s FOR UPDATE""",
                (str(code).upper(), participant_key),
            )
            row = cursor.fetchone()
        if row is None:  # pragma: no cover - defensive database boundary
            raise RuntimeError("challenge attempt commit was not persisted")
        stored_action = row["chosen_action_json"]
        if not isinstance(stored_action, str):
            stored_action = _canonical(stored_action)
        if stored_action != action_json:
            raise ChallengeAttemptConflictError(
                "participant already committed a different answer"
            )
        return {**self._attempt_row(row), "newlyCommitted": newly_committed}

    def list_attempts(self, code: str) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """SELECT * FROM challenge_locator_attempts_v1
                   WHERE code=%s ORDER BY created_at,attempt_id""",
                (str(code).upper(),),
            )
            rows = cursor.fetchall()
        return [self._attempt_row(row) for row in rows]


class ChallengeLocatorServiceV1:
    def __init__(
        self,
        store: SQLiteChallengeLocatorStore,
        *,
        rules: RuleBook,
        secret: bytes,
        public_base_url: str,
        clock: Callable[[], float] | None = None,
        difficulty_calibrator: Any | None = None,
        friend_ids: Callable[[str], Iterable[str]] | None = None,
        challenge_copy: Mapping[str, str] | None = None,
        research_presentation_resolver: (
            Callable[[str], Mapping[str, Any]] | None
        ) = None,
    ):
        if len(secret) < 32:
            raise ValueError("challenge locator secret must contain at least 32 bytes")
        base_url = str(public_base_url).strip().rstrip("/")
        if not base_url.startswith("https://") and not base_url.startswith("http://127.0.0.1"):
            raise ValueError("challenge locator requires HTTPS outside local development")
        self.store = store
        self.rules = rules
        self.secret = bytes(secret)
        self.public_base_url = base_url
        self.clock = clock or time.time
        self.difficulty_calibrator = difficulty_calibrator
        self.friend_ids = friend_ids
        self.challenge_copy = (
            {str(key): str(value) for key, value in challenge_copy.items()}
            if challenge_copy is not None else None
        )
        self.research_presentation_resolver = research_presentation_resolver

    def _research_presentation(self, result_code: str) -> dict[str, Any]:
        if self.research_presentation_resolver is None:
            raise ValueError(
                "Research challenge frozen presentation is unavailable"
            )
        raw = self.research_presentation_resolver(result_code)
        if not isinstance(raw, Mapping):
            raise ValueError("Research challenge frozen presentation is invalid")
        value = copy.deepcopy(dict(raw))
        expected = {
            "schemaVersion",
            "bundleRevisionId",
            "locale",
            "direction",
            "profileId",
            "profileVersion",
            "profileSha256",
            "challengeCopy",
        }
        challenge_copy = value.get("challengeCopy")
        if (
            set(value) != expected
            or value.get("schemaVersion") != 1
            or not isinstance(value.get("bundleRevisionId"), str)
            or not str(value["bundleRevisionId"]).strip()
            or value.get("locale") not in {"ar-SA", "en"}
            or value.get("direction") not in {"rtl", "ltr"}
            or not isinstance(value.get("profileId"), str)
            or not str(value["profileId"]).strip()
            or not isinstance(value.get("profileVersion"), str)
            or not str(value["profileVersion"]).strip()
            or _SHA256.fullmatch(str(value.get("profileSha256") or "")) is None
            or not isinstance(challenge_copy, Mapping)
        ):
            raise ValueError("Research challenge frozen presentation is invalid")
        normalized_copy = {
            str(key): str(copy_value)
            for key, copy_value in challenge_copy.items()
        }
        missing = sorted(
            key
            for key in _RESEARCH_CHALLENGE_COPY_KEYS
            if not normalized_copy.get(key, "").strip()
        )
        if missing:
            raise ValueError(
                "Research challenge frozen presentation lacks governed copy: "
                + ", ".join(missing)
            )
        copy_sha256 = hashlib.sha256(
            _canonical(normalized_copy).encode("utf-8")
        ).hexdigest()
        authority = {
            key: value[key]
            for key in (
                "schemaVersion",
                "bundleRevisionId",
                "locale",
                "direction",
                "profileId",
                "profileVersion",
                "profileSha256",
            )
        }
        authority["copySha256"] = copy_sha256
        return {
            "authority": authority,
            "copy": normalized_copy,
        }

    def _participant_key(self, code: str, participant_id: str) -> str:
        value = str(participant_id).strip()
        if not value:
            raise ValueError("challenge attempt requires a participant identifier")
        digest = hmac.new(
            self.secret,
            f"participant:{str(code).upper()}:{value}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"P-{digest[:32].upper()}"

    def _code(self, binding: Mapping[str, Any]) -> str:
        digest = hmac.new(self.secret, _canonical(binding).encode("utf-8"), hashlib.sha256).digest()
        return base64.b32encode(digest).decode("ascii").rstrip("=")[:16]

    def _research_binding_signature(self, binding: Mapping[str, Any]) -> str:
        unsigned = dict(binding)
        unsigned.pop("bindingSignature", None)
        return hmac.new(
            self.secret,
            f"research-result:{_canonical(unsigned)}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _public_projection(value: Mapping[str, Any]) -> dict[str, Any]:
        public = dict(value)
        public.pop("rulesHash", None)
        return public

    @staticmethod
    def _research_player_state(
        public_state: Mapping[str, Any], observer: str
    ) -> dict[str, Any]:
        state = copy.deepcopy(dict(public_state))
        hands = state.pop("hands", None)
        if not isinstance(hands, Mapping):
            raise ValueError("Research challenge hands are unavailable")
        observer_hand = hands.get(observer)
        if not isinstance(observer_hand, list):
            raise ValueError("Research challenge observer hand is unavailable")
        hand_sizes: list[int] = []
        for seat in _SEAT_ORDER:
            hand = hands.get(seat)
            if isinstance(hand, list):
                hand_sizes.append(len(hand))
            elif isinstance(hand, Mapping) and type(hand.get("count")) is int:
                hand_sizes.append(int(hand["count"]))
            else:
                raise ValueError("Research challenge hand size is invalid")
        plays: list[dict[str, Any]] = []
        for trick in state.pop("completedTricks", None) or ():
            if not isinstance(trick, Mapping):
                raise ValueError("Research challenge completed trick is invalid")
            plays.extend(
                copy.deepcopy([dict(item) for item in trick.get("plays") or ()])
            )
        current_trick = state.pop("currentTrick", None)
        if current_trick is not None:
            if not isinstance(current_trick, Mapping):
                raise ValueError("Research challenge current trick is invalid")
            plays.extend(
                copy.deepcopy([dict(item) for item in current_trick.get("plays") or ()])
            )
        state.update(
            {
                "observer": observer,
                "observerHand": copy.deepcopy(observer_hand),
                "knownUnplayed": {
                    seat: copy.deepcopy(observer_hand) if seat == observer else []
                    for seat in _SEAT_ORDER
                },
                "handSizes": hand_sizes,
                "plays": plays,
            }
        )
        return state

    @staticmethod
    def _research_player_action(
        action: Mapping[str, Any], decision_kind: str
    ) -> dict[str, Any]:
        if decision_kind != "play":
            return copy.deepcopy(dict(action))
        card = action.get("card")
        if not isinstance(card, Mapping):
            raise ValueError("Research challenge play action lacks its card")
        result = copy.deepcopy(dict(card))
        if str(result.get("code") or "") != str(action.get("code") or ""):
            raise ValueError("Research challenge play action code is inconsistent")
        if action.get("ekka") is not None:
            result["ekka"] = bool(action["ekka"])
        return result

    def publish(
        self,
        trace: CompleteGameTraceV1,
        *,
        decision_id: str,
        observer_seat: Seat | str,
        prompt_contract: str,
        reveal_policy: str = "after_commit",
        expires_at: int | None = None,
        answer_evidence: ChallengeAnswerEvidenceV1 | None = None,
        answer_explanation: str | None = None,
        player_names: Mapping[str, str] | None = None,
        rules: RuleBook | None = None,
    ) -> ChallengeLocatorV1:
        CompleteGameTraceV1.from_dict(trace.to_dict(trusted=True))
        observer = Seat.parse(observer_seat) if isinstance(observer_seat, str) else observer_seat
        replay_rules = rules or self.rules
        if replay_rules.sha256 != trace.rules_hash:
            raise ValueError("challenge locator rules do not match the frozen trace")
        store = TraceDecisionProjectionStoreV1(trace, replay_rules)
        record = next(
            (item for item in trace.decision_ledger.records if item.decision_id == decision_id),
            None,
        )
        if record is None:
            raise KeyError(decision_id)
        prompt = str(prompt_contract)
        if prompt == _ACTOR_PROMPT:
            if observer != record.acting_seat or record.forced:
                raise ValueError("acting-action challenge requires the non-forced acting-seat perspective")
        elif prompt in _OBSERVER_PROMPTS:
            if observer == record.acting_seat and prompt != "post_reveal_analysis":
                raise ValueError("non-acting prompt requires a non-acting observer")
        else:
            raise ValueError("unsupported challenge prompt contract")
        reveal = str(reveal_policy)
        if reveal not in {"after_commit", "after_completion", "post_reveal_only"}:
            raise ValueError("unsupported challenge reveal policy")
        expiry = int(expires_at) if expires_at is not None else None
        now = int(self.clock())
        if expiry is not None and expiry <= now:
            raise ValueError("challenge locator expiry must be in the future")
        projection = build_observer_projection(store, record.decision_id, observer)
        public_player_names: dict[str, str] | None = None
        if player_names is not None:
            public_player_names = {
                str(seat).strip().lower(): str(name).strip()
                for seat, name in player_names.items()
            }
            if set(public_player_names) != _PLAYER_NAME_SEATS:
                raise ValueError("challenge player names must contain exactly four seats")
            if any(not value or len(value) > 40 for value in public_player_names.values()):
                raise ValueError("challenge player names must contain 1 to 40 characters")
        if answer_evidence is not None:
            if prompt != _ACTOR_PROMPT:
                raise ValueError("governed answer evidence is currently actor-challenge only")
            if answer_evidence.trace_hash != trace.sha256:
                raise ValueError("challenge answer trace identity mismatch")
            if answer_evidence.decision_id != record.decision_id:
                raise ValueError("challenge answer decision identity mismatch")
            if answer_evidence.observer_seat != observer.name.lower():
                raise ValueError("challenge answer observer identity mismatch")
            if answer_evidence.projection_hash != projection.sha256:
                raise ValueError("challenge answer projection identity mismatch")
            legal_keys = {_canonical(dict(item)) for item in record.legal_actions}
            if _canonical(answer_evidence.recommendation) not in legal_keys:
                raise ValueError("challenge answer recommendation is not legal")
            if any(_canonical(item) not in legal_keys for item in answer_evidence.accepted_actions):
                raise ValueError("challenge answer accepted action is not legal")
        explanation = None if answer_explanation is None else str(answer_explanation).strip()
        if explanation is not None:
            if answer_evidence is None:
                raise ValueError("challenge answer explanation requires governed evidence")
            if not explanation or len(explanation) > 240 or "\n" in explanation:
                raise ValueError("challenge answer explanation must be one concise line")
        binding = {
            "schemaVersion": 1,
            "traceHash": trace.sha256,
            "decisionId": record.decision_id,
            "observerSeat": observer.name.lower(),
            "actingSeat": record.acting_seat.name.lower(),
            "promptContract": prompt,
            "revealPolicy": reveal,
            "projectionHash": projection.sha256,
            "sourceGameAction": dict(record.chosen_action),
            "answerEvidence": (
                answer_evidence.to_dict() if answer_evidence is not None else None
            ),
            "answerExplanation": explanation,
            "playerNames": public_player_names,
            "challengeCopy": self.challenge_copy,
            "expiresAt": expiry,
        }
        code = self._code(binding)
        public = {
            "schemaVersion": 1,
            "code": code,
            "observerSeat": observer.name.lower(),
            "actingSeat": record.acting_seat.name.lower(),
            "promptContract": prompt,
            "revealPolicy": reveal,
            "state": self._public_projection(projection.information_state),
            "legalActions": (
                [dict(item) for item in record.legal_actions]
                if prompt == _ACTOR_PROMPT else []
            ),
            "answerCommitted": False,
            "governedAnswerAvailable": answer_evidence is not None,
            "playerNames": public_player_names,
            "copy": self.challenge_copy,
            "expiresAt": expiry,
        }
        self.store.put(code, binding=binding, public=public, created_at=now, expires_at=expiry)
        return ChallengeLocatorV1(
            code,
            f"{self.public_base_url}/c/{code}",
            observer.name.lower(),
            record.acting_seat.name.lower(),
            prompt,
            reveal,
            expiry,
        )

    def publish_research_result(
        self,
        *,
        result_code: str,
        result_hash: str,
        exact_state_id: str,
        public_exact_state: Mapping[str, Any],
        proof_outcome: Mapping[str, Any],
        truth_label: str,
        confidence: float,
        reveal_policy: str = "after_commit",
    ) -> ChallengeLocatorV1:
        """Persist one stable, qualified Research result as a playable locator."""

        code = str(result_code).upper()
        digest = str(result_hash)
        state_id = str(exact_state_id).upper()
        presentation = self._research_presentation(code)
        frozen_copy = copy.deepcopy(dict(presentation["copy"]))
        presentation_authority = copy.deepcopy(dict(presentation["authority"]))
        if (
            _RESEARCH_RESULT_CODE.fullmatch(code) is None
            or _SHA256.fullmatch(digest) is None
            or code != f"LAF-{digest[:10].upper()}"
            or _RESEARCH_EXACT_STATE_ID.fullmatch(state_id) is None
        ):
            raise ValueError("Research challenge identity is invalid")
        exact = dict(public_exact_state)
        expected_keys = {
            "schemaVersion",
            "decisionReference",
            "publicProjectionHash",
            "phase",
            "decisionKind",
            "observerSeat",
            "nextActor",
            "legalActions",
            "publicState",
        }
        if set(exact) != expected_keys or exact.get("schemaVersion") != 1:
            raise ValueError("Research challenge public exact state is not closed")
        observer = str(exact.get("observerSeat") or "")
        actor = str(exact.get("nextActor") or "")
        reference = dict(exact.get("decisionReference") or {})
        projection_hash = str(exact.get("publicProjectionHash") or "")
        raw_legal = [dict(item) for item in exact.get("legalActions") or ()]
        if (
            exact.get("decisionKind") not in {"bid", "escalation", "project", "play"}
            or observer not in _PLAYER_NAME_SEATS
            or observer != actor
            or reference != {"resultCode": code, "decisionOrdinal": 1}
            or _SHA256.fullmatch(projection_hash) is None
            or not isinstance(exact.get("publicState"), Mapping)
            or not raw_legal
            or any(not isinstance(item, Mapping) for item in exact.get("legalActions") or ())
            or len({_canonical(item) for item in raw_legal}) != len(raw_legal)
        ):
            raise ValueError("Research challenge exact state is not playable")
        decision_kind = str(exact["decisionKind"])
        legal = [
            self._research_player_action(item, decision_kind) for item in raw_legal
        ]
        proof = dict(proof_outcome)
        if (
            proof.get("claimSatisfied") is not True
            or proof.get("completionStatus")
            not in {"exact_complete", "bounded_complete", "policy_complete"}
        ):
            raise ValueError("Research challenge proof is incomplete")
        selected_code = str(proof.get("selectedAction") or "")
        selected = next(
            (item for item in legal if str(item.get("code") or "") == selected_code),
            None,
        )
        roots = [dict(item) for item in proof.get("rootActions") or ()]
        roots_by_code = {
            str(item.get("action") or ""): item
            for item in roots
            if isinstance(item.get("action"), str)
        }
        legal_codes = {str(item.get("code") or "") for item in legal}
        if (
            set(roots_by_code) != legal_codes
            or any(
                type(item.get("satisfies")) is not bool
                or isinstance(item.get("value"), bool)
                or not isinstance(item.get("value"), (int, float))
                for item in roots_by_code.values()
            )
        ):
            raise ValueError("Research challenge proof root actions are invalid")
        accepted = [
            item
            for item in legal
            if dict(roots_by_code.get(str(item.get("code") or "")) or {}).get(
                "satisfies"
            )
            is True
        ]
        if selected is None or not accepted or selected not in accepted:
            raise ValueError("Research challenge proof does not bind a legal answer")
        ranked_legal = sorted(
            legal,
            key=lambda action: (
                0 if action == selected else 1,
                0
                if roots_by_code.get(str(action.get("code") or ""), {}).get(
                    "satisfies"
                )
                is True
                else 1,
                -float(
                    roots_by_code.get(str(action.get("code") or ""), {}).get(
                        "value", float("-inf")
                    )
                ),
                str(action.get("code") or ""),
            ),
        )
        answer_evidence = {
            "truthLabel": str(truth_label),
            "confidence": float(confidence),
            "answerMode": (
                "single_supported" if len(accepted) == 1 else "multiple_supported"
            ),
            "recommendation": selected,
            "acceptedActions": accepted,
            "actionEvidence": [
                {
                    "action": action,
                    "rank": index + 1,
                    "value": roots_by_code.get(str(action.get("code") or ""), {}).get(
                        "value"
                    ),
                    "satisfies": roots_by_code.get(
                        str(action.get("code") or ""), {}
                    ).get("satisfies")
                    is True,
                }
                for index, action in enumerate(ranked_legal)
            ],
            "engineCognitiveDifficulty": None,
            "proofBoundary": {
                "achievedMode": proof.get("achievedMode"),
                "completionStatus": proof.get("completionStatus"),
                "horizon": copy.deepcopy(proof.get("horizon")),
                "boundaryEvaluator": proof.get("boundaryEvaluator"),
            },
        }
        reveal = str(reveal_policy)
        if reveal not in {"after_commit", "after_completion"}:
            raise ValueError("Research challenge reveal policy is unsupported")
        public = {
            "schemaVersion": 1,
            "code": code,
            "observerSeat": observer,
            "actingSeat": actor,
            "decisionKind": decision_kind,
            "promptContract": _ACTOR_PROMPT,
            "revealPolicy": reveal,
            "state": self._research_player_state(exact["publicState"], observer),
            "legalActions": legal,
            "answerCommitted": False,
            "governedAnswerAvailable": True,
            "playerNames": None,
            "copy": frozen_copy,
            "copyProfile": presentation_authority,
            "expiresAt": None,
        }
        binding = {
            "schemaVersion": 1,
            "sourceKind": "qualified_research_result_v4",
            "resultCode": code,
            "resultHash": digest,
            "exactStateId": state_id,
            "publicProjectionHash": projection_hash,
            "observerSeat": observer,
            "actingSeat": actor,
            "promptContract": _ACTOR_PROMPT,
            "revealPolicy": reveal,
            "sourceGameAction": selected,
            "answerEvidence": answer_evidence,
            "answerExplanation": None,
            "challengeCopy": frozen_copy,
            "presentationAuthority": presentation_authority,
            "publicChallengeHash": hashlib.sha256(
                _canonical(public).encode("utf-8")
            ).hexdigest(),
            "expiresAt": None,
        }
        binding["bindingSignature"] = self._research_binding_signature(binding)
        self.store.put(
            code,
            binding=binding,
            public=public,
            created_at=int(self.clock()),
            expires_at=None,
        )
        return ChallengeLocatorV1(
            code,
            f"{self.public_base_url}/c/{code}",
            observer,
            actor,
            _ACTOR_PROMPT,
            reveal,
            None,
        )

    def _active_item(self, code: str) -> dict[str, Any]:
        item = self.store.get(str(code).upper())
        if item is None:
            raise KeyError(code)
        if item["revokedAt"] is not None:
            raise PermissionError("challenge locator has been revoked")
        if item["expiresAt"] is not None and int(self.clock()) >= item["expiresAt"]:
            raise PermissionError("challenge locator has expired")
        binding = dict(item["binding"])
        if binding.get("sourceKind") == "qualified_research_result_v4":
            signature = str(binding.get("bindingSignature") or "")
            if (
                item["code"] != binding.get("resultCode")
                or not hmac.compare_digest(
                    signature,
                    self._research_binding_signature(binding),
                )
                or not hmac.compare_digest(
                    str(binding.get("publicChallengeHash") or ""),
                    hashlib.sha256(
                        _canonical(item["public"]).encode("utf-8")
                    ).hexdigest(),
                )
            ):
                raise PermissionError("challenge locator integrity verification failed")
        elif not hmac.compare_digest(item["code"], self._code(binding)):
            raise PermissionError("challenge locator integrity verification failed")
        return item

    def resolve(self, code: str) -> dict[str, Any]:
        item = self._active_item(code)
        public = dict(item["public"])
        if item["binding"].get("challengeCopy") != public.get("copy"):
            raise PermissionError("challenge localization integrity verification failed")
        if item["binding"].get("sourceKind") == "qualified_research_result_v4":
            authority = item["binding"].get("presentationAuthority")
            if (
                authority != public.get("copyProfile")
                or not isinstance(authority, Mapping)
                or str(authority.get("copySha256") or "")
                != hashlib.sha256(
                    _canonical(dict(public.get("copy") or {})).encode("utf-8")
                ).hexdigest()
            ):
                raise PermissionError(
                    "challenge presentation integrity verification failed"
                )
        forbidden = {"traceHash", "decisionId", "projectionHash", "completeStateHash"}
        if forbidden.intersection(public):
            raise RuntimeError("public challenge locator leaked an internal identity")
        public["universalLink"] = f"{self.public_base_url}/c/{item['code']}"
        return public

    @staticmethod
    def _wilson_interval(successes: int, total: int) -> list[float]:
        if total <= 0:
            return [0.0, 0.0]
        z = 1.959963984540054
        proportion = successes / total
        denominator = 1.0 + z * z / total
        centre = (proportion + z * z / (2.0 * total)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1.0 - proportion) / total
                + z * z / (4.0 * total * total)
            )
            / denominator
        )
        return [round(max(0.0, centre - radius), 6), round(min(1.0, centre + radius), 6)]

    def _statistics(
        self,
        code: str,
        answer_evidence: Mapping[str, Any] | None,
        *,
        participant_id: str,
        include_friend_statistics: bool,
    ) -> dict[str, Any]:
        attempts = self.store.list_attempts(code)
        grouped: dict[str, dict[str, Any]] = {}
        for attempt in attempts:
            action = dict(attempt["chosenAction"])
            key = _canonical(action)
            grouped.setdefault(key, {"action": action, "count": 0})["count"] += 1
        total = len(attempts)
        choices = []
        evidence_by_action = {
            _canonical(dict(item["action"])): dict(item)
            for item in (answer_evidence or {}).get("actionEvidence") or ()
        }
        recommendation_key = (
            _canonical(dict(answer_evidence["recommendation"]))
            if answer_evidence is not None else None
        )
        recommended_count = 0
        for key in sorted(grouped):
            item = grouped[key]
            count = int(item["count"])
            if key == recommendation_key:
                recommended_count = count
            choice = {
                "action": item["action"],
                "count": count,
                "percentage": round(100.0 * count / total, 3) if total else 0.0,
                "proportionConfidence95": self._wilson_interval(count, total),
            }
            governed = evidence_by_action.get(key)
            if governed is not None:
                choice["evidence"] = {
                    name: value
                    for name, value in governed.items()
                    if name not in {"action", "rank"}
                }
                choice["rank"] = governed.get("rank")
            choices.append(choice)
        times = [
            float(item["responseSeconds"])
            for item in attempts
            if item.get("responseSeconds") is not None
        ]
        confidences = [
            float(item["confidence"])
            for item in attempts
            if item.get("confidence") is not None
        ]
        friends = (
            self._friend_statistics(code, participant_id, attempts)
            if include_friend_statistics
            else {"status": "authentication_required", "totalAttempts": 0, "choices": []}
        )
        audience = self._audience_difficulty(code, answer_evidence)
        return {
            "schemaVersion": 1,
            "totalAttempts": total,
            "distinctParticipants": len({item["participantKey"] for item in attempts}),
            "choices": choices,
            "medianResponseSeconds": (
                round(float(statistics.median(times)), 3) if times else None
            ),
            "meanConfidence": (
                round(sum(confidences) / len(confidences), 6)
                if confidences else None
            ),
            "changedAnswerRate": (
                round(sum(1 for item in attempts if item["changedAnswer"]) / total, 6)
                if total else None
            ),
            "recommendedActionPercentage": (
                round(100.0 * recommended_count / total, 3)
                if total and recommendation_key is not None else None
            ),
            "engineCognitiveDifficulty": (
                float(answer_evidence["engineCognitiveDifficulty"])
                if answer_evidence is not None
                and answer_evidence.get("engineCognitiveDifficulty") is not None
                else None
            ),
            "audienceDifficulty": audience,
            "friends": friends,
        }

    def _audience_difficulty(
        self,
        code: str,
        answer_evidence: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if answer_evidence is None:
            return {
                "status": "not_calibrated",
                "reason": "governed recommendation evidence is not attached",
            }
        if answer_evidence.get("engineCognitiveDifficulty") is None:
            return {
                "status": "not_calibrated",
                "reason": "governed engine difficulty evidence is not attached",
            }
        if self.difficulty_calibrator is None:
            return {
                "status": "not_calibrated",
                "reason": "audience calibrator is not configured",
            }
        try:
            item = self.difficulty_calibrator.recalibrate(
                code,
                engine_difficulty=float(answer_evidence["engineCognitiveDifficulty"]),
            )
        except Exception:
            return {
                "status": "unavailable",
                "reason": "audience calibration is temporarily unavailable",
            }
        result = {
            "status": "calibrated" if bool(item.get("public")) else "not_calibrated",
            "responseCount": int(item.get("responseCount", 0)),
            "distinctPlayers": int(item.get("distinctPlayers", 0)),
            "standardError": item.get("standardError"),
        }
        if bool(item.get("public")):
            result.update({
                "audienceCalibratedDifficulty": item.get("calibratedDifficulty"),
                "label": item.get("label"),
                "discrimination": item.get("discrimination"),
            })
        else:
            result["reason"] = "audience response thresholds have not been met"
        return result

    def _friend_statistics(
        self,
        code: str,
        participant_id: str,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.friend_ids is None:
            return {"status": "unavailable", "totalAttempts": 0, "choices": []}
        try:
            resolved_friends = tuple(self.friend_ids(participant_id))
        except Exception:
            return {"status": "unavailable", "totalAttempts": 0, "choices": []}
        friend_keys = {
            self._participant_key(code, friend_id)
            for friend_id in resolved_friends
            if str(friend_id).strip()
            and str(friend_id).strip() != str(participant_id).strip()
        }
        friend_attempts = [
            item for item in attempts if item["participantKey"] in friend_keys
        ]
        grouped: dict[str, dict[str, Any]] = {}
        for attempt in friend_attempts:
            action = dict(attempt["chosenAction"])
            key = _canonical(action)
            grouped.setdefault(key, {"action": action, "count": 0})["count"] += 1
        total = len(friend_attempts)
        return {
            "status": "available",
            "totalAttempts": total,
            "choices": [
                {
                    "action": grouped[key]["action"],
                    "count": int(grouped[key]["count"]),
                    "percentage": round(100.0 * grouped[key]["count"] / total, 3),
                }
                for key in sorted(grouped)
            ],
        }

    def commit_attempt(
        self,
        code: str,
        *,
        participant_id: str,
        chosen_action: Mapping[str, Any],
        response_seconds: float | None = None,
        confidence: float | None = None,
        changed_answer: bool = False,
        include_friend_statistics: bool = False,
        calibration_eligible: bool = False,
    ) -> dict[str, Any]:
        item = self._active_item(code)
        public = dict(item["public"])
        binding = dict(item["binding"])
        if public.get("promptContract") != _ACTOR_PROMPT:
            raise ValueError(
                "this prompt contract does not yet define a commit option schema"
            )
        if public.get("revealPolicy") == "post_reveal_only":
            raise ValueError("post-reveal-only challenges do not accept attempts")
        action = dict(chosen_action)
        legal = [dict(value) for value in public.get("legalActions") or ()]
        if _canonical(action) not in {_canonical(value) for value in legal}:
            raise ValueError("chosen action is not legal for this exact perspective state")
        if response_seconds is not None and not 0.0 <= float(response_seconds) <= 3600.0:
            raise ValueError("response seconds must be between zero and 3600")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between zero and one")
        participant_key = self._participant_key(item["code"], participant_id)
        attempt_digest = hmac.new(
            self.secret,
            f"attempt:{item['code']}:{participant_key}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        attempt = self.store.commit_attempt(
            attempt_id=f"A-{attempt_digest[:32].upper()}",
            code=item["code"],
            participant_key=participant_key,
            chosen_action=action,
            response_seconds=(
                None if response_seconds is None else float(response_seconds)
            ),
            confidence=None if confidence is None else float(confidence),
            changed_answer=bool(changed_answer),
            created_at=int(self.clock()),
        )
        source_action = dict(binding["sourceGameAction"])
        answer_evidence = binding.get("answerEvidence")
        reveal_allowed = public.get("revealPolicy") == "after_commit"
        chosen_key = _canonical(action)
        accepted_keys = {
            _canonical(dict(item))
            for item in (answer_evidence or {}).get("acceptedActions") or ()
        }
        calibration_record_failed = False
        if (
            bool(attempt.get("newlyCommitted"))
            and answer_evidence is not None
            and answer_evidence.get("engineCognitiveDifficulty") is not None
            and self.difficulty_calibrator is not None
            and bool(calibration_eligible)
        ):
            try:
                self.difficulty_calibrator.record_attempt(
                    decision_id=item["code"],
                    player_id=participant_id,
                    chosen_action=action,
                    accepted_actions=answer_evidence["acceptedActions"],
                    engine_difficulty=float(
                        answer_evidence["engineCognitiveDifficulty"]
                    ),
                    response_seconds=response_seconds,
                    confidence=confidence,
                    changed_answer=bool(changed_answer),
                    source="public_duplicate_challenge",
                )
            except Exception:
                # The committed challenge answer remains authoritative. Audience
                # calibration is derived evidence and may be reconciled later.
                calibration_record_failed = True
        if reveal_allowed and answer_evidence is not None:
            reveal = {
                "status": "governed_answer",
                "sourceGameAction": source_action,
                "matchedSourceGameAction": chosen_key == _canonical(source_action),
                "recommendationStatus": "governed_evidence",
                "truthLabel": answer_evidence["truthLabel"],
                "confidence": answer_evidence["confidence"],
                "answerMode": answer_evidence["answerMode"],
                "recommendedAction": answer_evidence["recommendation"],
                "acceptedActions": answer_evidence["acceptedActions"],
                "chosenAccepted": chosen_key in accepted_keys,
                "actionEvidence": answer_evidence["actionEvidence"],
                "engineCognitiveDifficulty": answer_evidence[
                    "engineCognitiveDifficulty"
                ],
                "proofBoundary": copy.deepcopy(answer_evidence.get("proofBoundary")),
                "explanation": binding.get("answerExplanation"),
            }
        elif reveal_allowed:
            reveal = {
                "status": "source_game_only",
                "sourceGameAction": source_action,
                "matchedSourceGameAction": chosen_key == _canonical(source_action),
                "recommendationStatus": "not_available",
                "truthLabel": "unresolved",
            }
        else:
            reveal = {
                "status": "withheld_until_completion",
                "recommendationStatus": (
                    "governed_evidence" if answer_evidence is not None else "not_available"
                ),
                "truthLabel": (
                    answer_evidence["truthLabel"]
                    if answer_evidence is not None else "unresolved"
                ),
            }
        statistics = self._statistics(
            item["code"],
            answer_evidence if reveal_allowed else None,
            participant_id=participant_id,
            include_friend_statistics=bool(include_friend_statistics),
        )
        if calibration_record_failed:
            statistics["audienceDifficulty"] = {
                "status": "unavailable",
                "reason": "the committed attempt requires calibration reconciliation",
            }
        return {
            "schemaVersion": 1,
            "attemptId": attempt["attemptId"],
            "code": item["code"],
            "committed": True,
            "chosenAction": action,
            "reveal": reveal,
            "statistics": statistics,
        }

    def restore_attempt(
        self,
        code: str,
        *,
        participant_id: str,
        include_friend_statistics: bool = False,
    ) -> dict[str, Any] | None:
        """Return the participant's immutable committed result after reload."""
        item = self._active_item(code)
        participant_key = self._participant_key(item["code"], participant_id)
        attempt = next(
            (
                candidate
                for candidate in self.store.list_attempts(item["code"])
                if candidate["participantKey"] == participant_key
            ),
            None,
        )
        if attempt is None:
            return None
        return self.commit_attempt(
            item["code"],
            participant_id=participant_id,
            chosen_action=attempt["chosenAction"],
            response_seconds=attempt["responseSeconds"],
            confidence=attempt["confidence"],
            changed_answer=attempt["changedAnswer"],
            include_friend_statistics=include_friend_statistics,
            calibration_eligible=False,
        )

    def revoke(self, code: str) -> bool:
        return self.store.revoke(str(code).upper(), revoked_at=int(self.clock()))

from __future__ import annotations

from typing import Any

from baloot_platform.shared.access import private_decision_review_allowed

from .arabic import explain_decision_ar


class ExplanationService:
    def __init__(self, learning_store: Any, audit_service: Any | None, config: dict[str, Any]):
        self.store = learning_store
        self.audit_service = audit_service
        self.config = dict(config)

    def explain(
        self,
        decision_id: str,
        *,
        observer: str | None = None,
        level: str = "basic",
        surface: str = "replay",
        run_completed: bool = False,
        challenge_revealed: bool = False,
        view_mode: str = "observer",
        ensure_audit: bool = False,
    ) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise KeyError(decision_id)
        can_audit = private_decision_review_allowed(
            actor=str(decision.get("actor") or ""),
            observer=observer or "",
            run_completed=run_completed,
        )
        audit = (decision.get("audits") or [None])[0] if can_audit else None
        if audit is None and ensure_audit and can_audit and self.audit_service is not None:
            try:
                audit = self.audit_service.audit_decision(decision_id, deep=False, force=False)
                decision = self.store.get_decision(decision_id) or decision
            except Exception:
                audit = None
        return explain_decision_ar(
            decision, audit=audit, observer=observer, level=level,
            surface=surface, run_completed=run_completed,
            challenge_revealed=challenge_revealed, view_mode=view_mode,
            config=self.config,
        ).to_dict()

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def submission_hash(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in dict(value).items()
        if key != "submissionHash"
    }
    return canonical_sha256(payload)


def validate_reviewer_attestation(
    reviewer: Mapping[str, Any],
) -> tuple[str, str]:
    value = dict(reviewer)
    reviewer_id = str(value.get("id", "")).strip()
    authority_id = str(value.get("authorityId", "")).strip()
    attestation = str(value.get("independenceAttestation", "")).strip()
    attestation_sha256 = str(
        value.get("independenceAttestationSha256", "")
    ).strip()
    if not reviewer_id or not authority_id or not attestation:
        raise ValueError("reviewer identity and attestation are required")
    expected = canonical_sha256(
        {
            "reviewerId": reviewer_id,
            "authorityId": authority_id,
            "attestation": attestation,
        }
    )
    if attestation_sha256 != expected:
        raise ValueError("reviewer independence attestation hash mismatch")
    return reviewer_id, authority_id


@dataclass(frozen=True, slots=True)
class AdjudicatedHoldout:
    dataset_id: str
    candidate_dataset_id: str
    candidate_dataset_sha256: str
    positions: tuple[dict[str, Any], ...]
    agreement_rate: float
    disagreements: int
    reviewer_ids: tuple[str, ...]
    authority_ids: tuple[str, ...]
    evidence_sha256: str
    exact_labels: int
    adjudicated_labels: int

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 1,
            "datasetId": self.dataset_id,
            "candidateDatasetId": self.candidate_dataset_id,
            "candidateDatasetSha256": self.candidate_dataset_sha256,
            "positionCount": len(self.positions),
            "agreementRate": self.agreement_rate,
            "disagreements": self.disagreements,
            "reviewerIds": list(self.reviewer_ids),
            "authorityIds": list(self.authority_ids),
            "adjudicationEvidenceSha256": self.evidence_sha256,
            "labelProvenance": {
                "exact": self.exact_labels,
                "independentlyAdjudicated": self.adjudicated_labels,
            },
            "positions": [dict(item) for item in self.positions],
            "automaticPromotion": False,
            "proofBoundary": (
                "Published labels come from deterministic local clone-minimax "
                "proof where tractable and agreement between distinct "
                "self-attested review authorities elsewhere. Integrity and "
                "coverage are proven; external identity, official rules "
                "certification, and human expertise are not inferred."
            ),
        }
        value["datasetSha256"] = canonical_sha256(value["positions"])
        value["manifestSha256"] = canonical_sha256(value)
        return value


class IndependentAdjudication:
    """Finalize a holdout only from complete independent review submissions."""

    def __init__(self, profile: Mapping[str, Any]):
        self.profile = dict(profile)

    @staticmethod
    def _reviewer(
        submission: Mapping[str, Any],
    ) -> tuple[str, str]:
        return validate_reviewer_attestation(
            dict(submission.get("reviewer") or {})
        )

    @staticmethod
    def _labels(
        submission: Mapping[str, Any],
    ) -> dict[str, tuple[str, float, dict[str, float]]]:
        labels: dict[str, tuple[str, float, dict[str, float]]] = {}
        for value in submission.get("labels") or ():
            label = dict(value)
            candidate_id = str(label.get("candidateId", "")).strip()
            action = str(label.get("action", "")).strip()
            confidence = float(label.get("confidence", math.nan))
            if not candidate_id or candidate_id in labels:
                raise ValueError(
                    "review labels require unique candidate identities"
                )
            if not action:
                raise ValueError("review label action is required")
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("review confidence must be in [0, 1]")
            raw_regrets = dict(label.get("actionRegrets") or {})
            regrets: dict[str, float] = {}
            for candidate_action, raw_regret in raw_regrets.items():
                key = str(candidate_action)
                regret = float(raw_regret)
                if (
                    not key
                    or not math.isfinite(regret)
                    or regret < 0.0
                ):
                    raise ValueError(
                        "review action regrets must be finite and nonnegative"
                    )
                regrets[key] = regret
            if (
                not regrets
                or min(regrets.values()) != 0.0
                or regrets.get(action) != 0.0
            ):
                raise ValueError(
                    "review best action must have zero regret"
                )
            labels[candidate_id] = (action, confidence, regrets)
        return labels

    def finalize(
        self,
        candidate_manifest: Mapping[str, Any],
        submissions: Iterable[Mapping[str, Any]],
        *,
        exact_evidence: Mapping[str, Any] | None = None,
    ) -> AdjudicatedHoldout:
        candidates = [
            dict(item)
            for item in candidate_manifest.get("positions") or ()
        ]
        candidate_id = str(
            candidate_manifest.get("datasetId", "")
        ).strip()
        candidate_sha256 = str(
            candidate_manifest.get("datasetSha256", "")
        ).strip()
        if not candidate_id or not candidate_sha256 or not candidates:
            raise ValueError("candidate corpus identity is incomplete")
        if canonical_sha256(candidates) != candidate_sha256:
            raise ValueError("candidate corpus hash mismatch")

        governance = dict(self.profile.get("permanent_holdout") or {})
        minimum_positions = int(governance.get("minimum_positions", 500))
        minimum_per_archetype = int(
            governance.get("minimum_per_archetype", 1)
        )
        minimum_agreement = float(
            governance.get("minimum_adjudication_agreement", 0.8)
        )
        minimum_authorities = int(
            governance.get("minimum_independent_authorities", 2)
        )
        require_full_coverage = bool(
            governance.get("require_full_candidate_coverage", True)
        )
        if minimum_authorities < 2:
            raise ValueError(
                "independent adjudication requires two authorities"
            )

        candidate_by_id = {
            str(item.get("candidateId", "")): item
            for item in candidates
        }
        if (
            "" in candidate_by_id
            or len(candidate_by_id) != len(candidates)
        ):
            raise ValueError("candidate identities must be unique")
        accepted: list[dict[str, Any]] = []
        exact_ids: set[str] = set()
        exact_hash = ""
        if exact_evidence is not None:
            exact = dict(exact_evidence)
            if (
                str(exact.get("candidateDatasetId", "")) != candidate_id
                or str(exact.get("candidateDatasetSha256", ""))
                != candidate_sha256
            ):
                raise ValueError("exact evidence candidate identity mismatch")
            exact_hash = str(exact.get("evidenceSha256", ""))
            exact_payload = {
                key: value
                for key, value in exact.items()
                if key != "evidenceSha256"
            }
            if exact_hash != canonical_sha256(exact_payload):
                raise ValueError("exact evidence hash mismatch")
            oracle_id = str(exact.get("oracleId", "")).strip()
            oracle_sha256 = str(
                exact.get("oracleAuthoritySha256", "")
            ).strip()
            if (
                not oracle_id
                or len(oracle_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in oracle_sha256.lower()
                )
            ):
                raise ValueError("exact oracle identity is invalid")
            exact_positions = [
                dict(item) for item in exact.get("positions") or ()
            ]
            if int(exact.get("exactLabelCount", -1)) != len(
                exact_positions
            ):
                raise ValueError("exact evidence count mismatch")
            for label in exact_positions:
                position_id = str(label.get("candidateId", "")).strip()
                if (
                    not position_id
                    or position_id in exact_ids
                    or position_id not in candidate_by_id
                ):
                    raise ValueError("exact label identity is invalid")
                candidate = candidate_by_id[position_id]
                exact_ids.add(position_id)
                if (
                    str(label.get("stateSha256", ""))
                    != str(candidate["stateSha256"])
                    or str(label.get("observer", ""))
                    != str(candidate["observer"])
                    or list(label.get("legalActions") or ())
                    != list(candidate["legalActions"])
                ):
                    raise ValueError("exact label source state mismatch")
                legal = {
                    str(value) for value in candidate["legalActions"]
                }
                action_values = [
                    dict(item) for item in label.get("actionValues") or ()
                ]
                values = {
                    str(item.get("action", "")): float(
                        item.get("expectedTeamMargin", math.nan)
                    )
                    for item in action_values
                }
                if (
                    len(action_values) != len(legal)
                    or set(values) != legal
                    or any(not math.isfinite(value) for value in values.values())
                ):
                    raise ValueError("exact label action values are invalid")
                best = max(values.values())
                references = sorted(
                    str(value)
                    for value in label.get("referenceActions") or ()
                )
                expected = sorted(
                    action
                    for action, value in values.items()
                    if value == best
                )
                if references != expected:
                    raise ValueError("exact optimal action set mismatch")
                if (
                    str(label.get("labelSource", "")) != "exact"
                    or str(label.get("searchCompletion", "")) != "exact"
                    or str(label.get("labelAuthorityId", "")) != oracle_id
                    or str(label.get("labelAuthoritySha256", ""))
                    != oracle_sha256
                    or int(label.get("nodes", -1)) < 0
                ):
                    raise ValueError("exact label provenance is invalid")
                accepted.append(
                    {
                        "positionId": position_id,
                        "split": "permanent_holdout",
                        "stateSha256": candidate["stateSha256"],
                        "rulesSha256": candidate["rulesSha256"],
                        "configurationSha256": (
                            candidate["configurationSha256"]
                        ),
                        "engineVersion": candidate["engineVersion"],
                        "observer": candidate["observer"],
                        "observerState": candidate["observerState"],
                        "dealHash": candidate["dealHash"],
                        "episodeId": candidate["episodeId"],
                        "sourceGroup": candidate["sourceGroup"],
                        "seedGroup": candidate["seedGroup"],
                        "archetype": candidate["archetype"],
                        "legalActions": list(candidate["legalActions"]),
                        "referenceAction": references[0],
                        "referenceActions": references,
                        "actionRegrets": {
                            action: best - value
                            for action, value in sorted(values.items())
                        },
                        "actionValues": [
                            dict(item)
                            for item in label["actionValues"]
                        ],
                        "labelSource": "exact",
                        "labelAuthorityId": label["labelAuthorityId"],
                        "labelAuthoritySha256": (
                            label["labelAuthoritySha256"]
                        ),
                        "searchCompletion": "exact",
                        "oracleNodes": int(label.get("nodes", 0)),
                    }
                )
        remaining_ids = set(candidate_by_id) - exact_ids
        submissions_list = [dict(item) for item in submissions]
        if remaining_ids and len(submissions_list) < minimum_authorities:
            raise ValueError("insufficient independent review submissions")

        reviewer_ids: list[str] = []
        authority_ids: list[str] = []
        labels_by_reviewer: list[
            dict[str, tuple[str, float, dict[str, float]]]
        ] = []
        verified_submissions: list[dict[str, Any]] = []
        for submission in submissions_list:
            if str(submission.get("datasetId", "")) != candidate_id:
                raise ValueError("review submission dataset identity mismatch")
            if (
                str(submission.get("datasetSha256", ""))
                != candidate_sha256
            ):
                raise ValueError("review submission dataset hash mismatch")
            if str(submission.get("submissionHash", "")) != submission_hash(
                submission
            ):
                raise ValueError("review submission hash mismatch")
            reviewer_id, authority_id = self._reviewer(submission)
            reviewer_ids.append(reviewer_id)
            authority_ids.append(authority_id)
            labels = self._labels(submission)
            if require_full_coverage and not remaining_ids.issubset(labels):
                raise ValueError(
                    "review submission must cover the complete corpus of "
                    "all non-exact candidates"
                )
            unknown = set(labels) - set(candidate_by_id)
            if unknown:
                raise ValueError("review submission contains unknown positions")
            for position_id, (action, _, regrets) in labels.items():
                legal = {
                    str(value)
                    for value in candidate_by_id[position_id].get(
                        "legalActions"
                    )
                    or ()
                }
                if action not in legal:
                    raise ValueError(
                        "review submission selected an illegal action"
                    )
                if set(regrets) != legal:
                    raise ValueError(
                        "review action regrets must cover every legal action"
                    )
            labels_by_reviewer.append(labels)
            verified_submissions.append(submission)
        if len(set(reviewer_ids)) != len(reviewer_ids):
            raise ValueError("reviewers must be distinct")
        if len(set(authority_ids)) != len(authority_ids):
            raise ValueError("review authorities must be distinct")

        evidence_sha256 = canonical_sha256(
            {
                "exactEvidenceSha256": exact_hash or None,
                "reviewSubmissionSha256": [
                    str(item["submissionHash"])
                    for item in sorted(
                        verified_submissions,
                        key=lambda item: str(item["submissionHash"]),
                    )
                ],
            }
        )
        disagreements = 0
        shared_ids = (
            set.intersection(
                *(set(labels) for labels in labels_by_reviewer)
            )
            if labels_by_reviewer
            else set()
        )
        adjudicated = 0
        for position_id in sorted(remaining_ids & shared_ids):
            candidate = candidate_by_id[position_id]
            decisions = [
                labels[position_id] for labels in labels_by_reviewer
            ]
            actions = {decision[0] for decision in decisions}
            if len(actions) != 1:
                disagreements += 1
                continue
            action = decisions[0][0]
            action_regrets = {
                legal_action: statistics.fmean(
                    decision[2][legal_action]
                    for decision in decisions
                )
                for legal_action in candidate["legalActions"]
            }
            reference_actions = sorted(
                legal_action
                for legal_action, regret in action_regrets.items()
                if regret == 0.0
            )
            accepted.append(
                {
                    "positionId": position_id,
                    "split": "permanent_holdout",
                    "stateSha256": candidate["stateSha256"],
                    "rulesSha256": candidate["rulesSha256"],
                    "configurationSha256": (
                        candidate["configurationSha256"]
                    ),
                    "engineVersion": candidate["engineVersion"],
                    "observer": candidate["observer"],
                    "observerState": candidate["observerState"],
                    "dealHash": candidate["dealHash"],
                    "episodeId": candidate["episodeId"],
                    "sourceGroup": candidate["sourceGroup"],
                    "seedGroup": candidate["seedGroup"],
                    "archetype": candidate["archetype"],
                    "legalActions": list(candidate["legalActions"]),
                    "referenceAction": reference_actions[0],
                    "referenceActions": reference_actions,
                    "actionRegrets": action_regrets,
                    "labelSource": "independent_adjudication",
                    "labelAuthorityId": (
                        f"independent-review-{evidence_sha256[:20]}"
                    ),
                    "labelAuthoritySha256": evidence_sha256,
                    "adjudicatorCount": len(reviewer_ids),
                    "adjudicators": sorted(reviewer_ids),
                    "adjudicationAgreement": 1.0,
                    "labelConfidence": min(
                        decision[1] for decision in decisions
                    ),
                }
            )
            adjudicated += 1

        disagreements = len(remaining_ids) - adjudicated
        agreement_rate = (
            adjudicated / len(remaining_ids) if remaining_ids else 1.0
        )
        if agreement_rate < minimum_agreement:
            raise ValueError("independent review agreement is below policy")
        if len(accepted) < minimum_positions:
            raise ValueError("too few agreed labels for permanent holdout")
        counts = Counter(
            str(item["archetype"]) for item in accepted
        )
        missing = [
            str(archetype)
            for archetype in self.profile.get("strata") or ()
            if counts[str(archetype)] < minimum_per_archetype
        ]
        if missing:
            raise ValueError(
                "agreed labels lack archetype coverage: "
                + ", ".join(sorted(missing))
            )
        accepted.sort(key=lambda item: str(item["positionId"]))
        dataset_id = (
            "permanent-holdout-"
            f"{canonical_sha256(accepted)[:20]}"
        )
        return AdjudicatedHoldout(
            dataset_id=dataset_id,
            candidate_dataset_id=candidate_id,
            candidate_dataset_sha256=candidate_sha256,
            positions=tuple(accepted),
            agreement_rate=agreement_rate,
            disagreements=disagreements,
            reviewer_ids=tuple(sorted(reviewer_ids)),
            authority_ids=tuple(sorted(authority_ids)),
            evidence_sha256=evidence_sha256,
            exact_labels=len(exact_ids),
            adjudicated_labels=adjudicated,
        )

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def certification_approval_payload(value: Mapping[str, Any]) -> bytes:
    """Canonical bytes signed by an externally governed rules approver."""

    schema_version = int(value.get("schemaVersion", 1))
    payload = {
        "schemaVersion": schema_version,
        "purpose": "balootx_rules_certification",
        "profileId": value.get("profileId"),
        "profileVersion": value.get("profileVersion"),
        "profileHash": value.get("profileHash"),
        "goldenCases": value.get("goldenCases", []),
        "adjudicationProfileHash": value.get("adjudicationProfileHash"),
    }
    if schema_version >= 2:
        payload["sourceCatalog"] = value.get("sourceCatalog")
    return _canonical(payload)


@dataclass(frozen=True, slots=True)
class RulesCertificationCandidateV18:
    candidate_id: str
    profile_id: str
    profile_version: str
    profile_hash: str
    golden_cases: tuple[Mapping[str, Any], ...]
    adjudication_profile_hash: str
    source_catalog_identity: Mapping[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    external_signatures: tuple[Mapping[str, Any], ...] = ()
    candidate_hash: str = ""

    @classmethod
    def create(cls, *, profile_id: str, profile_version: str, profile: Mapping[str, Any],
               golden_cases: Sequence[Mapping[str, Any]], adjudication_profile: Mapping[str, Any],
               external_signatures: Sequence[Mapping[str, Any]] = (),
               source_catalog_identity: Mapping[str, Any] | None = None) -> "RulesCertificationCandidateV18":
        profile_hash = _hash(profile)
        adjudication_hash = _hash(adjudication_profile)
        cases = tuple(dict(x) for x in golden_cases)
        payload = {
            "profileId": profile_id, "profileVersion": profile_version, "profileHash": profile_hash,
            "goldenCases": list(cases), "adjudicationProfileHash": adjudication_hash,
            "externalSignatures": list(external_signatures),
        }
        source_catalog = (
            dict(source_catalog_identity) if source_catalog_identity is not None else None
        )
        if source_catalog is not None:
            payload["sourceCatalog"] = source_catalog
        candidate_hash = _hash(payload)
        return cls(
            candidate_id=f"RULECERT-{candidate_hash[:20].upper()}", profile_id=profile_id,
            profile_version=profile_version, profile_hash=profile_hash, golden_cases=cases,
            adjudication_profile_hash=adjudication_hash,
            source_catalog_identity=source_catalog,
            external_signatures=tuple(dict(x) for x in external_signatures), candidate_hash=candidate_hash,
        )

    @property
    def official(self) -> bool:
        # A candidate cannot establish its own trust. Official status is only
        # returned by verify_certification_candidate with configured keys.
        return False

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": 2 if self.source_catalog_identity is not None else 1,
            "candidateId": self.candidate_id, "profileId": self.profile_id,
            "profileVersion": self.profile_version, "profileHash": self.profile_hash,
            "goldenCases": [dict(x) for x in self.golden_cases],
            "adjudicationProfileHash": self.adjudication_profile_hash, "createdAt": self.created_at,
            "externalSignatures": [dict(x) for x in self.external_signatures],
            "official": self.official, "candidateHash": self.candidate_hash,
            "automaticRuleActivation": False,
        }
        if self.source_catalog_identity is not None:
            value["sourceCatalog"] = dict(self.source_catalog_identity)
        return value


def build_certification_candidate(*, rules_path: str | Path, adjudication_path: str | Path,
                                  golden_cases_path: str | Path,
                                  external_signatures: Sequence[Mapping[str, Any]] = (),
                                  source_catalog_identity: Mapping[str, Any] | None = None) -> RulesCertificationCandidateV18:
    import yaml
    rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
    adjudication = yaml.safe_load(Path(adjudication_path).read_text(encoding="utf-8"))
    cases_value = json.loads(Path(golden_cases_path).read_text(encoding="utf-8"))
    cases = cases_value.get("cases", cases_value) if isinstance(cases_value, dict) else cases_value
    profile = rules.get("profile", {})
    return RulesCertificationCandidateV18.create(
        profile_id=str(profile.get("id", "rules")), profile_version=str(profile.get("version", "0")),
        profile=rules, golden_cases=cases, adjudication_profile=adjudication,
        external_signatures=external_signatures,
        source_catalog_identity=source_catalog_identity,
    )


def verify_certification_candidate(
    value: Mapping[str, Any],
    *,
    trusted_signers: Sequence[Mapping[str, Any]] = (),
    required_signer_roles: Sequence[str] = (
        "certified_referee",
        "rules_authority",
    ),
    minimum_external_signatures: int = 2,
    expected_profile_identity: Mapping[str, Any] | None = None,
    expected_source_catalog_identity: Mapping[str, Any] | None = None,
    expected_adjudication_profile_hash: str | None = None,
    certification_eligible: bool = True,
) -> dict[str, Any]:
    schema_version = int(value.get("schemaVersion", 1))
    payload = {
        "profileId": value.get("profileId"), "profileVersion": value.get("profileVersion"),
        "profileHash": value.get("profileHash"), "goldenCases": value.get("goldenCases", []),
        "adjudicationProfileHash": value.get("adjudicationProfileHash"),
        "externalSignatures": value.get("externalSignatures", []),
    }
    if schema_version >= 2:
        payload["sourceCatalog"] = value.get("sourceCatalog")
    actual = _hash(payload)
    signatures = list(value.get("externalSignatures", []))
    trusted = {
        str(item.get("signer_id", "")): dict(item)
        for item in trusted_signers
        if item.get("signer_id")
    }
    approval_payload = certification_approval_payload(value)
    verified_roles: set[str] = set()
    verified_signers: list[str] = []
    signature_results: list[dict[str, Any]] = []
    seen_signers: set[str] = set()
    for item in signatures:
        signer_id = str(item.get("signerId", ""))
        role = str(item.get("role", ""))
        valid = False
        reason = "untrusted_signer"
        signer = trusted.get(signer_id)
        if signer_id in seen_signers:
            reason = "duplicate_signer"
        elif signer is not None and role != str(signer.get("role", "")):
            reason = "role_mismatch"
        elif signer is not None and item.get("algorithm") != "ed25519":
            reason = "unsupported_algorithm"
        elif signer is not None:
            try:
                from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                    Ed25519PublicKey,
                )

                key = Ed25519PublicKey.from_public_bytes(
                    base64.b64decode(str(signer["public_key_base64"]), validate=True)
                )
                key.verify(
                    base64.b64decode(str(item.get("signature", "")), validate=True),
                    approval_payload,
                )
                valid = True
                reason = "verified"
            except Exception:
                reason = "signature_invalid"
        seen_signers.add(signer_id)
        if valid:
            verified_signers.append(signer_id)
            verified_roles.add(role)
        signature_results.append(
            {"signerId": signer_id, "role": role, "valid": valid, "reason": reason}
        )
    hash_valid = actual == value.get("candidateHash")
    required_roles = {str(role) for role in required_signer_roles}
    signatures_valid = (
        len(verified_signers) >= int(minimum_external_signatures)
        and required_roles <= verified_roles
    )
    profile_identity_valid = expected_profile_identity is None or {
        "id": value.get("profileId"),
        "version": value.get("profileVersion"),
        "sha256": value.get("profileHash"),
    } == dict(expected_profile_identity)
    expected_catalog = dict(expected_source_catalog_identity or {})
    source_catalog = dict(value.get("sourceCatalog") or {})
    source_catalog_identity_valid = expected_source_catalog_identity is None or all(
        source_catalog.get(key) == expected_catalog.get(key)
        for key in ("id", "version", "sha256")
    )
    adjudication_profile_identity_valid = (
        expected_adjudication_profile_hash is None
        or value.get("adjudicationProfileHash")
        == expected_adjudication_profile_hash
    )
    authority_identity_valid = (
        profile_identity_valid
        and source_catalog_identity_valid
        and adjudication_profile_identity_valid
    )
    official = (
        bool(certification_eligible)
        and hash_valid
        and signatures_valid
        and authority_identity_valid
    )
    return {
        "valid": hash_valid,
        "expectedHash": value.get("candidateHash"),
        "actualHash": actual,
        "official": official,
        "certificationEligible": bool(certification_eligible),
        "signaturesValid": signatures_valid,
        "externalSignerCount": len(signatures),
        "verifiedSignerCount": len(verified_signers),
        "verifiedSignerIds": verified_signers,
        "verifiedRoles": sorted(verified_roles),
        "signatureResults": signature_results,
        "profileIdentityValid": profile_identity_valid,
        "sourceCatalogIdentityValid": source_catalog_identity_valid,
        "adjudicationProfileIdentityValid": adjudication_profile_identity_valid,
        "authorityIdentityValid": authority_identity_valid,
        "automaticRuleActivation": False,
    }

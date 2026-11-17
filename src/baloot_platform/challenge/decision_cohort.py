"""Actor-observer seed authority for deterministic decision cohorts.

The authority deliberately excludes generation and result identity.  In
particular, root/attempt seeds, game or trace identifiers, decision IDs, and
selected actions cannot influence the derived cohort seed.
"""
from __future__ import annotations

from typing import Final

from baloot_platform.core.enums import Seat

from .evidence_v3 import canonical_sha256


_SEED_POLICY: Final = "actor_observer_cohort_seed_v1"
_SEED_SCHEMA_VERSION: Final = 1
_DECISION_KINDS: Final = frozenset({"bid", "play", "project"})
_SEATS: Final = frozenset(seat.name.lower() for seat in Seat)
_MAX_COHORT_INDEX: Final = (1 << 31) - 1
_MAX_GOVERNED_SALT_LENGTH: Final = 256
PRIMARY_DECISION_COHORT_SALT_V1: Final = "primary-decision-policy:v1"


def _sha256_hex(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a canonical lowercase SHA-256 digest")
    return value


def _decision_kind(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("decision_kind must be a string")
    if value not in _DECISION_KINDS:
        raise ValueError(f"decision_kind must be one of {sorted(_DECISION_KINDS)}")
    return value


def _acting_seat(value: Seat | str) -> str:
    if isinstance(value, Seat):
        return value.name.lower()
    if not isinstance(value, str):
        raise TypeError("acting_seat must be a Seat or canonical seat string")
    if value not in _SEATS:
        raise ValueError(f"acting_seat must be one of {sorted(_SEATS)}")
    return value


def _cohort_index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("cohort_index must be an integer")
    if not 0 <= value <= _MAX_COHORT_INDEX:
        raise ValueError(f"cohort_index must be in [0, {_MAX_COHORT_INDEX}]")
    return value


def _governed_salt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("governed_salt must be a string")
    if not value or value != value.strip():
        raise ValueError("governed_salt must be a non-empty canonical string")
    if len(value) > _MAX_GOVERNED_SALT_LENGTH:
        raise ValueError(
            f"governed_salt must not exceed {_MAX_GOVERNED_SALT_LENGTH} characters"
        )
    if not value.isprintable():
        raise ValueError("governed_salt must contain only printable characters")
    return value


def derive_actor_observer_cohort_seed_v1(
    *,
    public_prefix_hash: str,
    decision_kind: str,
    acting_seat: Seat | str,
    cohort_authority_hash: str,
    cohort_index: int,
    governed_salt: str,
) -> int:
    """Return a stable unsigned 64-bit seed for one actor-visible cohort.

    Only the declared actor-observer state and governed cohort authority enter
    the canonical preimage.  Hidden deals and generation/result identifiers
    are intentionally neither accepted nor inferred.
    """

    payload = {
        "actingSeat": _acting_seat(acting_seat),
        "cohortAuthorityHash": _sha256_hex(
            cohort_authority_hash, field="cohort_authority_hash"
        ),
        "cohortIndex": _cohort_index(cohort_index),
        "decisionKind": _decision_kind(decision_kind),
        "governedSalt": _governed_salt(governed_salt),
        "policy": _SEED_POLICY,
        "publicPrefixHash": _sha256_hex(public_prefix_hash, field="public_prefix_hash"),
        "schemaVersion": _SEED_SCHEMA_VERSION,
    }
    return int(canonical_sha256(payload)[:16], 16)

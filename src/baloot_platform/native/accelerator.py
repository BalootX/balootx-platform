from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from baloot_platform.core.cards import Card, iter_mask
from baloot_platform.core.enums import Contract, Rank, Seat, Team
from baloot_platform.core.models import PlayedCard, Project, RoundState
from baloot_platform.core.play_ir import PlayDomainIRV1
from baloot_platform.core.project_ir import ProjectDomainIRV1
from baloot_platform.core.projects import announce_project_for_turn, score_project_team
from baloot_platform.core.rules import RuleBook
from baloot_platform.core.scoring_ir import ScoringDomainIRV1


class NativeUnavailable(RuntimeError):
    """Raised when the optional native accelerator cannot be loaded safely."""


# ABI V1 is retained for downstream compatibility.
class _RulesV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("sun_strength", ctypes.c_int32 * 8),
        ("hokum_non_trump_strength", ctypes.c_int32 * 8),
        ("hokum_trump_strength", ctypes.c_int32 * 8),
        ("allow_discard_when_cannot_overtrump", ctypes.c_uint8),
        ("partner_winning_exception", ctypes.c_uint8),
        ("locked_hokum_blocks_trump_lead", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _PlayV1(ctypes.Structure):
    _fields_ = [
        ("seat", ctypes.c_uint8),
        ("card", ctypes.c_uint8),
        ("ekka", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _StateV1(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("hands", ctypes.c_uint32 * 4),
        ("contract", ctypes.c_uint8),
        ("trump_suit", ctypes.c_int8),
        ("current_player", ctypes.c_uint8),
        ("locked", ctypes.c_uint8),
        ("trick_size", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("trick", _PlayV1 * 3),
    ]


_StrengthTable = ctypes.c_int32 * 8
_StrengthTables = _StrengthTable * 3
_PointTable = ctypes.c_int16 * 8
_PointTables = _PointTable * 3
_ProjectValues = ctypes.c_uint16 * 3
_FourKindValues = ctypes.c_uint16 * 8


class _RulesV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("strengths", _StrengthTables),
        ("points", _PointTables),
        ("sequence_values_sun", _ProjectValues),
        ("sequence_values_hokum", _ProjectValues),
        ("four_kind_values_sun", _FourKindValues),
        ("four_kind_values_hokum", _FourKindValues),
        ("last_trick_bonus_raw", ctypes.c_uint16),
        ("baloot_game_points", ctypes.c_uint16),
        ("kabout_sun", ctypes.c_uint16),
        ("kabout_hokum", ctypes.c_uint16),
        ("match_target", ctypes.c_uint16),
        ("must_follow_suit", ctypes.c_uint8),
        ("must_trump_when_void", ctypes.c_uint8),
        ("must_overtrump_when_possible", ctypes.c_uint8),
        ("allow_discard_when_cannot_overtrump", ctypes.c_uint8),
        ("partner_winning_exception", ctypes.c_uint8),
        ("locked_hokum_blocks_trump_lead", ctypes.c_uint8),
        ("ekka_enabled", ctypes.c_uint8),
        ("projects_enabled", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 8),
    ]


class _PlayV2(ctypes.Structure):
    _fields_ = [
        ("seat", ctypes.c_uint8),
        ("card", ctypes.c_uint8),
        ("ekka", ctypes.c_uint8),
        ("baloot", ctypes.c_uint8),
    ]


class _StateV2(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("hands", ctypes.c_uint32 * 4),
        ("initial_hands", ctypes.c_uint32 * 4),
        ("contract", ctypes.c_uint8),
        ("trump_suit", ctypes.c_int8),
        ("current_player", ctypes.c_uint8),
        ("locked", ctypes.c_uint8),
        ("trump_broken", ctypes.c_uint8),
        ("buyer", ctypes.c_uint8),
        ("dealer", ctypes.c_uint8),
        ("first_leader", ctypes.c_uint8),
        ("trick_size", ctypes.c_uint8),
        ("completed_tricks", ctypes.c_uint8),
        ("escalation_multiplier", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8 * 2),
        ("raw_points", ctypes.c_int32 * 2),
        ("project_points", ctypes.c_int32 * 2),
        ("baloot_points", ctypes.c_int32 * 2),
        ("match_score", ctypes.c_int32 * 2),
        ("trick", _PlayV2 * 3),
        ("reserved1", ctypes.c_uint8 * 4),
    ]


class _TrickResultV2(ctypes.Structure):
    _fields_ = [("winner", ctypes.c_int32), ("raw_points", ctypes.c_int32)]


class _UndoV2(ctypes.Structure):
    _fields_ = [("previous", _StateV2)]


class _RulesV3(ctypes.Structure):
    _fields_ = [
        ("base", _RulesV2),
        ("card_game_point_totals", ctypes.c_uint16 * 2),
        ("project_multiplier_cap", ctypes.c_uint8),
        ("baloot_multiplier_cap", ctypes.c_uint8),
        ("kabout_enabled", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 5),
    ]


class _StateV3(ctypes.Structure):
    _fields_ = [
        ("base", _StateV2),
        ("declarer", ctypes.c_uint8),
        ("counting_team", ctypes.c_uint8),
        ("escalation_mode", ctypes.c_uint8),
        ("tricks_won", ctypes.c_uint8 * 2),
        ("baloot_eligible_seats_mask", ctypes.c_uint8),
        ("baloot_declared_teams_mask", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 1),
    ]


class _SearchLimitsV3(ctypes.Structure):
    _fields_ = [
        ("maximum_nodes", ctypes.c_uint64),
        ("maximum_milliseconds", ctypes.c_uint32),
        ("maximum_cache_entries", ctypes.c_uint32),
        ("maximum_depth", ctypes.c_uint8),
        ("team", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 6),
    ]


class _SearchResultV3(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_uint64),
        ("value", ctypes.c_int32),
        ("own_points", ctypes.c_int32),
        ("opponent_points", ctypes.c_int32),
        ("score_after_round", ctypes.c_int32 * 2),
        ("match_winner", ctypes.c_int8),
        ("exact", ctypes.c_uint8),
        ("status", ctypes.c_uint8),
        ("principal_variation_length", ctypes.c_uint8),
        ("contract_succeeded", ctypes.c_uint8),
        ("round_lead", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
        ("principal_variation", ctypes.c_uint8 * 32),
    ]


class _RolloutResultV3(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_uint64),
        ("value", ctypes.c_int32),
        ("own_points", ctypes.c_int32),
        ("opponent_points", ctypes.c_int32),
        ("score_after_round", ctypes.c_int32 * 2),
        ("match_winner", ctypes.c_int8),
        ("exact", ctypes.c_uint8),
        ("status", ctypes.c_uint8),
        ("contract_succeeded", ctypes.c_uint8),
        ("round_lead", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _RolloutTaskV3(ctypes.Structure):
    _fields_ = [
        ("state", _StateV3),
        ("initial_card", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


_ROLLOUT_NO_INITIAL_CARD_V3 = 255


@dataclass(frozen=True, slots=True)
class NativeTrickResult:
    winner: Seat
    raw_points: int


@dataclass(frozen=True, slots=True)
class NativeStateSummary:
    legal_moves_mask: int
    state_hash: int
    raw_points: tuple[int, int]
    current_player: Seat
    completed_tricks: int


@dataclass(frozen=True, slots=True)
class NativeExactSearchResult:
    value: float
    own_points: float
    opponent_points: float
    exact: bool
    nodes: int
    principal_variation: tuple[str, ...]
    contract_succeeded: bool
    round_lead: bool
    score_after_round: tuple[int, int]
    match_winner: Team | None
    completion_status: str


@dataclass(frozen=True, slots=True)
class NativeRolloutSummary:
    value: float
    own_points: float
    opponent_points: float
    exact: bool
    nodes: int
    contract_succeeded: bool
    round_lead: bool
    score_after_round: tuple[int, int]
    match_winner: Team | None
    completion_status: str


_CONTRACT_IDS = {Contract.SUN: 0, Contract.HOKUM: 1, Contract.ASHKAL: 2}
_RANK_INDEX = {
    "seven": int(Rank.SEVEN),
    "eight": int(Rank.EIGHT),
    "nine": int(Rank.NINE),
    "ten": int(Rank.TEN),
    "jack": int(Rank.JACK),
    "queen": int(Rank.QUEEN),
    "king": int(Rank.KING),
    "ace": int(Rank.ACE),
}
_ESCALATION_MULTIPLIER = {"normal": 1, "double": 2, "triple": 3, "four": 4, "gahwa": 1}
_ESCALATION_MODE = {"normal": 0, "double": 1, "triple": 2, "four": 3, "gahwa": 4}
_NONE_SEAT = 255


class NativeAccelerator:
    """Versioned C++20 accelerator backed by the canonical rules profile.

    The Python reference engine remains the semantic oracle. V2 accelerates
    legal-action batches, card points, trick evaluation, deterministic state
    hashing and apply-play snapshots. Structure sizes and capabilities are
    verified at load time to prevent silent cross-language memory corruption.
    """

    CAP_LEGAL_MOVES = 1
    CAP_BATCH_LEGAL_MOVES = 2
    CAP_CARD_POINTS = 4
    CAP_TRICK_EVALUATION = 8
    CAP_STATE_HASH = 16
    CAP_APPLY_PLAY = 32
    CAP_APPLY_UNDO = 64
    CAP_EXACT_SEARCH = 128

    def __init__(self, rules: RuleBook, library: str | Path | None = None):
        library_path = self._locate_library(library)
        try:
            self.lib = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise NativeUnavailable(
                "native accelerator could not be loaded for "
                f"{sys.platform}: {library_path}"
            ) from exc
        self._configure_v1()
        self.rules = rules
        self.play_domain = rules.play_domain.require_integrity(
            rules.structural_domain
        )
        self.project_domain = rules.project_domain
        self.scoring_domain = ScoringDomainIRV1.from_rules(rules.data)
        self.rules_v1 = self._build_rules_v1(rules, self.play_domain)
        self.supports_v2 = self._configure_v2_if_available()
        self.rules_v2 = (
            self._build_rules_v2(
                rules,
                self.project_domain,
                self.scoring_domain,
                self.play_domain,
            )
            if self.supports_v2
            else None
        )
        self.supports_v3 = self._configure_v3_if_available()
        self.rules_v3 = (
            self._build_rules_v3(
                rules,
                self.project_domain,
                self.scoring_domain,
                self.play_domain,
            )
            if self.supports_v3
            else None
        )
        self.supports_compact_rollout = self._configure_compact_rollout_if_available()

    @staticmethod
    def _locate_library(library: str | Path | None) -> Path:
        explicit = str(library or os.environ.get("BALOOT_NATIVE_LIBRARY", "")).strip()
        if explicit:
            candidate = Path(explicit)
            if candidate.is_file():
                return candidate
            raise NativeUnavailable(f"native accelerator library does not exist: {candidate}")
        root = Path(__file__).resolve().parents[3]
        candidates = [root / item for item in NativeAccelerator._library_candidates()]
        found = next((item for item in candidates if item.is_file()), None)
        if found is None:
            raise NativeUnavailable("native accelerator library was not found")
        return found

    @staticmethod
    def _library_candidates(platform_name: str | None = None) -> tuple[str, ...]:
        platform_name = platform_name or sys.platform
        if platform_name == "win32":
            return (
                "build/native-msvc/Release/baloot_native.dll",
                "build/native/Release/baloot_native.dll",
                "build/native-release/Release/baloot_native.dll",
            )
        if platform_name == "darwin":
            return (
                "build/native/libbaloot_native.dylib",
                "build/native-release/libbaloot_native.dylib",
            )
        return (
            "build/native-release/libbaloot_native.so",
            "build/native/libbaloot_native.so",
        )

    def _configure_v1(self) -> None:
        self.lib.baloot_native_abi_version.restype = ctypes.c_uint32
        if int(self.lib.baloot_native_abi_version()) != 1:
            raise NativeUnavailable("native accelerator ABI V1 mismatch")
        self.lib.baloot_legal_moves_v1.argtypes = [ctypes.POINTER(_RulesV1), ctypes.POINTER(_StateV1)]
        self.lib.baloot_legal_moves_v1.restype = ctypes.c_uint32
        self.lib.baloot_trick_winner_v1.argtypes = [
            ctypes.POINTER(_RulesV1),
            ctypes.c_uint8,
            ctypes.c_int8,
            ctypes.POINTER(_PlayV1),
        ]
        self.lib.baloot_trick_winner_v1.restype = ctypes.c_int32

    def _configure_v2_if_available(self) -> bool:
        try:
            abi = self.lib.baloot_native_abi_version_v2
        except AttributeError:
            self.capabilities = 0
            return False
        abi.restype = ctypes.c_uint32
        if int(abi()) != 2:
            raise NativeUnavailable("native accelerator ABI V2 mismatch")
        for name, py_type in (
            ("baloot_sizeof_rules_v2", _RulesV2),
            ("baloot_sizeof_state_v2", _StateV2),
            ("baloot_sizeof_play_v2", _PlayV2),
            ("baloot_sizeof_undo_v2", _UndoV2),
        ):
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_uint32
            native_size = int(fn())
            python_size = ctypes.sizeof(py_type)
            if native_size != python_size:
                raise NativeUnavailable(
                    f"native ABI structure mismatch for {py_type.__name__}: native={native_size}, python={python_size}"
                )
        self.lib.baloot_native_capabilities_v2.restype = ctypes.c_uint32
        self.capabilities = int(self.lib.baloot_native_capabilities_v2())
        required = (
            self.CAP_LEGAL_MOVES
            | self.CAP_BATCH_LEGAL_MOVES
            | self.CAP_CARD_POINTS
            | self.CAP_TRICK_EVALUATION
            | self.CAP_STATE_HASH
            | self.CAP_APPLY_PLAY
            | self.CAP_APPLY_UNDO
        )
        if self.capabilities & required != required:
            raise NativeUnavailable(f"native V2 capabilities incomplete: {self.capabilities:#x}")
        self.lib.baloot_legal_moves_v2.argtypes = [ctypes.POINTER(_RulesV2), ctypes.POINTER(_StateV2)]
        self.lib.baloot_legal_moves_v2.restype = ctypes.c_uint32
        self.lib.baloot_legal_moves_batch_v2.argtypes = [
            ctypes.POINTER(_RulesV2),
            ctypes.POINTER(_StateV2),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self.lib.baloot_legal_moves_batch_v2.restype = ctypes.c_int32
        self.lib.baloot_card_points_v2.argtypes = [
            ctypes.POINTER(_RulesV2),
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_int8,
        ]
        self.lib.baloot_card_points_v2.restype = ctypes.c_int32
        self.lib.baloot_evaluate_trick_v2.argtypes = [
            ctypes.POINTER(_RulesV2),
            ctypes.c_uint8,
            ctypes.c_int8,
            ctypes.POINTER(_PlayV2),
            ctypes.POINTER(_TrickResultV2),
        ]
        self.lib.baloot_evaluate_trick_v2.restype = ctypes.c_int32
        self.lib.baloot_state_hash_v2.argtypes = [ctypes.POINTER(_StateV2)]
        self.lib.baloot_state_hash_v2.restype = ctypes.c_uint64
        self.lib.baloot_apply_play_v2.argtypes = [
            ctypes.POINTER(_RulesV2),
            ctypes.POINTER(_StateV2),
            ctypes.c_uint8,
            ctypes.c_uint8,
            ctypes.c_uint8,
        ]
        self.lib.baloot_apply_play_v2.restype = ctypes.c_int32
        self.lib.baloot_apply_play_with_undo_v2.argtypes = [
            ctypes.POINTER(_RulesV2), ctypes.POINTER(_StateV2), ctypes.c_uint8,
            ctypes.c_uint8, ctypes.c_uint8, ctypes.POINTER(_UndoV2),
        ]
        self.lib.baloot_apply_play_with_undo_v2.restype = ctypes.c_int32
        self.lib.baloot_undo_play_v2.argtypes = [ctypes.POINTER(_StateV2), ctypes.POINTER(_UndoV2)]
        self.lib.baloot_undo_play_v2.restype = ctypes.c_int32
        return True

    def _configure_v3_if_available(self) -> bool:
        if not self.supports_v2:
            return False
        try:
            abi = self.lib.baloot_native_abi_version_v3
        except AttributeError:
            return False
        abi.restype = ctypes.c_uint32
        if int(abi()) != 3:
            raise NativeUnavailable("native accelerator ABI V3 mismatch")
        for name, py_type in (
            ("baloot_sizeof_rules_v3", _RulesV3),
            ("baloot_sizeof_state_v3", _StateV3),
            ("baloot_sizeof_search_limits_v3", _SearchLimitsV3),
            ("baloot_sizeof_search_result_v3", _SearchResultV3),
            ("baloot_sizeof_rollout_task_v3", _RolloutTaskV3),
        ):
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_uint32
            native_size = int(fn())
            python_size = ctypes.sizeof(py_type)
            if native_size != python_size:
                raise NativeUnavailable(
                    f"native ABI structure mismatch for {py_type.__name__}: "
                    f"native={native_size}, python={python_size}"
                )
        self.lib.baloot_exact_search_v3.argtypes = [
            ctypes.POINTER(_RulesV3),
            ctypes.POINTER(_StateV3),
            ctypes.POINTER(_SearchLimitsV3),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_SearchResultV3),
        ]
        self.lib.baloot_exact_search_v3.restype = ctypes.c_int32
        self.lib.baloot_heuristic_rollout_batch_v3.argtypes = [
            ctypes.POINTER(_RulesV3),
            ctypes.POINTER(_RolloutTaskV3),
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_SearchResultV3),
        ]
        self.lib.baloot_heuristic_rollout_batch_v3.restype = ctypes.c_int32
        return True

    def _configure_compact_rollout_if_available(self) -> bool:
        if not self.supports_v3:
            return False
        try:
            size_fn = self.lib.baloot_sizeof_rollout_result_v3
            batch_fn = self.lib.baloot_heuristic_rollout_batch_compact_v3
        except AttributeError:
            return False
        size_fn.restype = ctypes.c_uint32
        native_size = int(size_fn())
        python_size = ctypes.sizeof(_RolloutResultV3)
        if native_size != python_size:
            raise NativeUnavailable(
                "native ABI structure mismatch for _RolloutResultV3: "
                f"native={native_size}, python={python_size}"
            )
        batch_fn.argtypes = [
            ctypes.POINTER(_RulesV3),
            ctypes.POINTER(_RolloutTaskV3),
            ctypes.c_uint32,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_RolloutResultV3),
        ]
        batch_fn.restype = ctypes.c_int32
        return True

    @staticmethod
    def _build_rules_v1(
        rules: RuleBook,
        play_domain: PlayDomainIRV1 | None = None,
    ) -> _RulesV1:
        domain = (play_domain or rules.play_domain).require_integrity(
            rules.structural_domain
        )
        return _RulesV1(
            1,
            (ctypes.c_int32 * 8)(*domain.sun_strength),
            (ctypes.c_int32 * 8)(*domain.hokum_non_trump_strength),
            (ctypes.c_int32 * 8)(*domain.hokum_trump_strength),
            int(domain.allow_discard_when_cannot_overtrump),
            int(
                domain.partner_winning_exception
                == "position_aware_ace_or_ekka"
            ),
            int(domain.locked_hokum_blocks_trump_lead),
            0,
        )

    @staticmethod
    def _build_rules_v2(
        rules: RuleBook,
        project_domain: ProjectDomainIRV1 | None = None,
        scoring_domain: ScoringDomainIRV1 | None = None,
        play_domain: PlayDomainIRV1 | None = None,
    ) -> _RulesV2:
        project_domain = project_domain or rules.project_domain
        scoring_domain = scoring_domain or ScoringDomainIRV1.from_rules(rules.data)
        if scoring_domain.buyer_tie_outcome is not None:
            raise NativeUnavailable("native scoring ABI does not support configured Sun buyer tie outcomes")
        return NativeAccelerator._build_transition_rules_v2(rules, project_domain, scoring_domain, play_domain)

    @staticmethod
    def _build_transition_rules_v2(rules, project_domain, scoring_domain, play_domain=None) -> _RulesV2:
        """Play tables only; DD supplies scoring from the engine, not ABI V3."""
        play_domain = (play_domain or rules.play_domain).require_integrity(
            rules.structural_domain
        )
        native = _RulesV2()
        native.abi_version = 2
        strength_tables = (
            play_domain.sun_strength,
            play_domain.hokum_non_trump_strength,
            play_domain.hokum_trump_strength,
        )
        point_tables = (
            play_domain.sun_points,
            play_domain.hokum_non_trump_points,
            play_domain.hokum_trump_points,
        )
        for table_index in range(3):
            for rank_index in range(8):
                native.strengths[table_index][rank_index] = int(strength_tables[table_index][rank_index])
                native.points[table_index][rank_index] = int(point_tables[table_index][rank_index])
        for index, key in enumerate(("three", "four", "five_plus")):
            native.sequence_values_sun[index] = project_domain.sequence_value_for(
                Contract.SUN, key
            )
            native.sequence_values_hokum[index] = project_domain.sequence_value_for(
                Contract.HOKUM, key
            )
        for name, index in _RANK_INDEX.items():
            native.four_kind_values_sun[index] = project_domain.four_kind_value_for(
                Contract.SUN, name
            )
            native.four_kind_values_hokum[index] = project_domain.four_kind_value_for(
                Contract.HOKUM, name
            )
        native.last_trick_bonus_raw = scoring_domain.last_trick_bonus_raw
        native.baloot_game_points = project_domain.baloot_value
        native.kabout_sun = scoring_domain.kabout_base_game_points["sun"]
        native.kabout_hokum = scoring_domain.kabout_base_game_points["hokum"]
        native.match_target = scoring_domain.match_target
        native.must_follow_suit = int(play_domain.must_follow_suit)
        native.must_trump_when_void = int(play_domain.must_trump_when_void)
        native.must_overtrump_when_possible = int(
            play_domain.must_overtrump_when_possible
        )
        native.allow_discard_when_cannot_overtrump = int(
            play_domain.allow_discard_when_cannot_overtrump
        )
        native.partner_winning_exception = int(
            play_domain.partner_winning_exception
            == "position_aware_ace_or_ekka"
        )
        native.locked_hokum_blocks_trump_lead = int(
            play_domain.locked_hokum_blocks_trump_lead
        )
        native.ekka_enabled = int(play_domain.ekka_enabled)
        native.projects_enabled = int(project_domain.enabled)
        return native

    @staticmethod
    def _build_rules_v3(
        rules: RuleBook,
        project_domain: ProjectDomainIRV1 | None = None,
        scoring_domain: ScoringDomainIRV1 | None = None,
        play_domain: PlayDomainIRV1 | None = None,
    ) -> _RulesV3:
        project_domain = project_domain or rules.project_domain
        scoring_domain = scoring_domain or ScoringDomainIRV1.from_rules(rules.data)
        play_domain = (play_domain or rules.play_domain).require_integrity(
            rules.structural_domain
        )
        native = _RulesV3()
        native.base = NativeAccelerator._build_rules_v2(
            rules, project_domain, scoring_domain, play_domain
        )
        native.card_game_point_totals[0] = scoring_domain.card_game_point_totals["sun"]
        native.card_game_point_totals[1] = scoring_domain.card_game_point_totals["hokum"]
        native.project_multiplier_cap = scoring_domain.project_multiplier_cap
        native.baloot_multiplier_cap = scoring_domain.baloot_multiplier_cap
        native.kabout_enabled = int(scoring_domain.kabout_enabled)
        return native

    @staticmethod
    def _contract_id(contract: Contract) -> int:
        return _CONTRACT_IDS[Contract.parse(contract)]

    @staticmethod
    def _trump_id(state: RoundState) -> int:
        return int(state.trump_suit) if state.trump_suit is not None else -1

    @staticmethod
    def _seat_id(seat: Seat | None) -> int:
        return int(seat) if seat is not None else _NONE_SEAT

    def _resolved_projects(self, state: RoundState) -> list[Project]:
        if not state.projects or any(project.scoring for project in state.projects):
            return state.projects
        if self.project_domain.tie_break_order == "second_trick_reveal" and not state.completed_tricks:
            raise ValueError("native project projection requires the first trick winner")
        # Legacy declarations are play-independent; reveal-order comparison
        # additionally requires the recorded first-trick winner.
        resolved = list(state.projects)
        first = state.first_leader
        if first is None:
            raise ValueError("project-bearing state is missing its first leader")
        for offset in range(4):
            seat = Seat((int(first) + offset) % 4)
            resolved, _announced = announce_project_for_turn(
                resolved,
                seat,
                trump_suit=state.trump_suit,
                domain=self.project_domain,
            )
        resolved, _winner = score_project_team(
            resolved,
            trump_suit=state.trump_suit,
            domain=self.project_domain,
            second_trick_leader=(
                state.completed_tricks[0].winner if state.completed_tricks else None
            ),
        )
        return resolved

    def _project_points(self, state: RoundState) -> tuple[int, int]:
        return self.project_domain.project_points(self._resolved_projects(state))

    def _baloot_points(self, state: RoundState) -> tuple[int, int]:
        values = [0, 0]
        for team in state.baloot_teams:
            values[int(team)] += self.project_domain.baloot_value
        return values[0], values[1]

    @staticmethod
    def _raw_points(state: RoundState) -> tuple[int, int]:
        values = [0, 0]
        for trick in state.completed_tricks:
            if trick.winner is not None:
                values[int(trick.winner.team)] += int(trick.raw_points)
        return values[0], values[1]

    def _state_v1(self, state: RoundState) -> _StateV1:
        native = _StateV1()
        native.abi_version = 1
        for index, hand in enumerate(state.hands):
            native.hands[index] = int(hand)
        assert state.contract is not None and state.current_player is not None and state.current_trick is not None
        native.contract = self._contract_id(state.contract)
        native.trump_suit = self._trump_id(state)
        native.current_player = int(state.current_player)
        native.locked = int(state.locked)
        native.trick_size = len(state.current_trick.plays)
        for index, play in enumerate(state.current_trick.plays[:3]):
            native.trick[index] = _PlayV1(int(play.seat), play.card.id, int(play.ekka), 0)
        return native

    def _state_v2(self, state: RoundState) -> _StateV2:
        if state.current_player is None or state.contract is None or state.current_trick is None:
            raise ValueError("a playing RoundState is required")
        native = _StateV2()
        native.abi_version = 2
        for index in range(4):
            native.hands[index] = int(state.hands[index])
            native.initial_hands[index] = int(state.initial_hands[index] if state.initial_hands else 0)
        native.contract = self._contract_id(state.contract)
        native.trump_suit = self._trump_id(state)
        native.current_player = int(state.current_player)
        native.locked = int(state.locked)
        native.trump_broken = int(state.trump_broken)
        native.buyer = self._seat_id(state.buyer)
        native.dealer = self._seat_id(state.dealer)
        native.first_leader = self._seat_id(state.first_leader)
        native.trick_size = len(state.current_trick.plays)
        native.completed_tricks = len(state.completed_tricks)
        native.escalation_multiplier = _ESCALATION_MULTIPLIER.get(state.escalation, 1)
        raw_points = self._raw_points(state)
        project_points = self._project_points(state)
        baloot_points = self._baloot_points(state)
        for index in range(2):
            native.raw_points[index] = raw_points[index]
            native.project_points[index] = project_points[index]
            native.baloot_points[index] = baloot_points[index]
            native.match_score[index] = int(state.match_score[index])
        for index, play in enumerate(state.current_trick.plays[:3]):
            native.trick[index] = _PlayV2(int(play.seat), play.card.id, int(play.ekka), int(play.baloot))
        return native

    @staticmethod
    def _counting_team(state: RoundState) -> Team:
        if state.declarer is None:
            raise ValueError("native exact search requires a declarer")
        if state.escalation == "normal":
            return state.declarer.team.other
        if state.escalation_history:
            return Team[str(state.escalation_history[-1]["team"]).upper()].other
        return state.declarer.team.other

    def _baloot_eligible_seats_mask(self, state: RoundState) -> int:
        if state.contract != Contract.HOKUM or state.trump_suit is None:
            return 0
        hands = {
            seat: tuple(iter_mask(state.initial_hands[int(seat)]))
            for seat in Seat
        }
        eligible = self.project_domain.baloot_eligible_seats(
            hands,
            self._resolved_projects(state),
            state.contract,
            state.trump_suit,
        )
        return sum(1 << int(seat) for seat in eligible)

    def _state_v3(self, state: RoundState) -> _StateV3:
        if state.declarer is None:
            raise ValueError("native exact search requires a declarer")
        native = _StateV3()
        native.base = self._state_v2(state)
        native.declarer = int(state.declarer)
        native.counting_team = int(self._counting_team(state))
        native.escalation_mode = _ESCALATION_MODE[state.escalation]
        tricks_won = [0, 0]
        for trick in state.completed_tricks:
            if trick.winner is None:
                raise ValueError("completed trick is missing a winner")
            tricks_won[int(trick.winner.team)] += 1
        native.tricks_won[0] = tricks_won[0]
        native.tricks_won[1] = tricks_won[1]
        native.baloot_eligible_seats_mask = self._baloot_eligible_seats_mask(state)
        declared = 0
        for team in state.baloot_teams:
            declared |= 1 << int(team)
        native.baloot_declared_teams_mask = declared
        return native

    @staticmethod
    def _playing_state(state: RoundState) -> bool:
        return state.current_player is not None and state.contract is not None and state.current_trick is not None

    def legal_moves_mask(self, state: RoundState) -> int:
        if not self._playing_state(state):
            return 0
        if self.supports_v2:
            assert self.rules_v2 is not None
            native = self._state_v2(state)
            return int(self.lib.baloot_legal_moves_v2(ctypes.byref(self.rules_v2), ctypes.byref(native)))
        native_v1 = self._state_v1(state)
        return int(self.lib.baloot_legal_moves_v1(ctypes.byref(self.rules_v1), ctypes.byref(native_v1)))

    def legal_moves(self, state: RoundState) -> tuple[Card, ...]:
        return tuple(iter_mask(self.legal_moves_mask(state)))

    def legal_moves_masks(self, states: Sequence[RoundState]) -> tuple[int, ...]:
        if not states:
            return ()
        if not self.supports_v2:
            return tuple(self.legal_moves_mask(state) for state in states)
        assert self.rules_v2 is not None
        native_states = (_StateV2 * len(states))(*(self._state_v2(state) for state in states))
        output = (ctypes.c_uint32 * len(states))()
        result = int(
            self.lib.baloot_legal_moves_batch_v2(
                ctypes.byref(self.rules_v2), native_states, len(states), output
            )
        )
        if result != len(states):
            raise NativeUnavailable(f"native batch legal-move evaluation failed: {result}")
        return tuple(int(output[index]) for index in range(len(states)))

    def card_points(self, card: Card | int, contract: Contract, trump_suit: int | None = None) -> int:
        card = Card.parse(card)
        if not self.supports_v2:
            trump = None if trump_suit is None or int(trump_suit) < 0 else trump_suit
            return self.rules.points(card, Contract.parse(contract), trump)  # type: ignore[arg-type]
        assert self.rules_v2 is not None
        result = int(
            self.lib.baloot_card_points_v2(
                ctypes.byref(self.rules_v2),
                card.id,
                self._contract_id(Contract.parse(contract)),
                -1 if trump_suit is None else int(trump_suit),
            )
        )
        if result < 0:
            raise NativeUnavailable(f"native card point evaluation failed: {result}")
        return result

    def evaluate_trick(
        self,
        plays: Sequence[PlayedCard | tuple[int | Seat, Card | int]],
        contract: Contract,
        trump_suit: int | None,
    ) -> NativeTrickResult:
        if len(plays) != 4:
            raise ValueError("exactly four plays are required")
        normalized: list[_PlayV2] = []
        for item in plays:
            if isinstance(item, PlayedCard):
                normalized.append(_PlayV2(int(item.seat), item.card.id, int(item.ekka), int(item.baloot)))
            else:
                seat, card = item
                normalized.append(_PlayV2(int(Seat.parse(seat)), Card.parse(card).id, 0, 0))
        if not self.supports_v2:
            v1_plays = (_PlayV1 * 4)(*(_PlayV1(p.seat, p.card, p.ekka, 0) for p in normalized))
            winner = int(
                self.lib.baloot_trick_winner_v1(
                    ctypes.byref(self.rules_v1),
                    self._contract_id(Contract.parse(contract)),
                    -1 if trump_suit is None else int(trump_suit),
                    v1_plays,
                )
            )
            points = sum(self.card_points(p.card, contract, trump_suit) for p in normalized)
            return NativeTrickResult(Seat(winner), points)
        assert self.rules_v2 is not None
        native_plays = (_PlayV2 * 4)(*normalized)
        result = _TrickResultV2()
        status = int(
            self.lib.baloot_evaluate_trick_v2(
                ctypes.byref(self.rules_v2),
                self._contract_id(Contract.parse(contract)),
                -1 if trump_suit is None else int(trump_suit),
                native_plays,
                ctypes.byref(result),
            )
        )
        if status != 0 or result.winner not in range(4):
            raise NativeUnavailable(f"native trick evaluation failed: {status}")
        return NativeTrickResult(Seat(result.winner), int(result.raw_points))

    def state_hash(self, state: RoundState) -> int:
        if not self.supports_v2:
            raise NativeUnavailable("state hashing requires native ABI V2")
        assert self.rules_v2 is not None
        native = self._state_v2(state)
        return int(self.lib.baloot_state_hash_v2(ctypes.byref(native)))

    def apply_undo_roundtrip(
        self, state: RoundState, card: Card | int, *, ekka: bool = False, baloot: bool = False,
    ) -> dict[str, int | bool]:
        """Apply then undo one play in native memory and prove exact restoration."""
        if not self.supports_v2 or not (self.capabilities & self.CAP_APPLY_UNDO):
            raise NativeUnavailable("apply/undo requires native ABI V2 undo capability")
        assert self.rules_v2 is not None
        native = self._state_v2(state)
        before = int(self.lib.baloot_state_hash_v2(ctypes.byref(native)))
        undo = _UndoV2()
        status = int(self.lib.baloot_apply_play_with_undo_v2(
            ctypes.byref(self.rules_v2), ctypes.byref(native), Card.parse(card).id,
            int(ekka), int(baloot), ctypes.byref(undo),
        ))
        if status < 0:
            raise ValueError(f"native play rejected with status {status}")
        applied = int(self.lib.baloot_state_hash_v2(ctypes.byref(native)))
        undone = int(self.lib.baloot_undo_play_v2(ctypes.byref(native), ctypes.byref(undo)))
        if undone != 0:
            raise NativeUnavailable(f"native undo failed: {undone}")
        after = int(self.lib.baloot_state_hash_v2(ctypes.byref(native)))
        return {"status": status, "beforeHash": before, "appliedHash": applied,
                "afterHash": after, "restored": before == after}

    def apply_play_summary(
        self,
        state: RoundState,
        card: Card | int,
        *,
        ekka: bool = False,
        baloot: bool = False,
    ) -> NativeStateSummary:
        """Apply a play to an isolated native snapshot without mutating Python state."""
        if not self.supports_v2:
            raise NativeUnavailable("apply-play requires native ABI V2")
        assert self.rules_v2 is not None
        native = self._state_v2(state)
        status = int(
            self.lib.baloot_apply_play_v2(
                ctypes.byref(self.rules_v2),
                ctypes.byref(native),
                Card.parse(card).id,
                int(ekka),
                int(baloot),
            )
        )
        if status < 0:
            raise ValueError(f"native play rejected with status {status}")
        legal = int(self.lib.baloot_legal_moves_v2(ctypes.byref(self.rules_v2), ctypes.byref(native)))
        return NativeStateSummary(
            legal_moves_mask=legal,
            state_hash=int(self.lib.baloot_state_hash_v2(ctypes.byref(native))),
            raw_points=(int(native.raw_points[0]), int(native.raw_points[1])),
            current_player=Seat(int(native.current_player)),
            completed_tricks=int(native.completed_tricks),
        )

    @staticmethod
    def _start_cancellation_poller(
        cancel_check: Callable[[], bool] | None,
    ) -> tuple[ctypes.c_uint32, threading.Event, threading.Thread | None]:
        cancelled = ctypes.c_uint32(0)
        stop_polling = threading.Event()
        poller: threading.Thread | None = None
        if cancel_check is not None:
            if bool(cancel_check()):
                cancelled.value = 1
            else:
                def poll_cancel() -> None:
                    while not stop_polling.wait(0.01):
                        if bool(cancel_check()):
                            cancelled.value = 1
                            return

                poller = threading.Thread(
                    target=poll_cancel,
                    name="baloot-native-search-cancellation",
                    daemon=True,
                )
                poller.start()
        return cancelled, stop_polling, poller

    @staticmethod
    def _stop_cancellation_poller(
        stop_polling: threading.Event,
        poller: threading.Thread | None,
    ) -> None:
        stop_polling.set()
        if poller is not None:
            poller.join(timeout=0.1)

    @staticmethod
    def _decode_search_result(result: _SearchResultV3) -> NativeExactSearchResult:
        completion = {
            0: "exact",
            1: "budget_exhausted",
            2: "cancelled",
            3: "failed",
            4: "heuristic_complete",
        }.get(int(result.status))
        if completion is None:
            raise NativeUnavailable(
                f"native search returned an unknown status: {result.status}"
            )
        line_length = int(result.principal_variation_length)
        if not 0 <= line_length <= 32:
            raise NativeUnavailable("native search returned an invalid line length")
        match_winner = (
            None if int(result.match_winner) < 0 else Team(int(result.match_winner))
        )
        return NativeExactSearchResult(
            value=float(result.value),
            own_points=float(result.own_points),
            opponent_points=float(result.opponent_points),
            exact=bool(result.exact),
            nodes=int(result.nodes),
            principal_variation=tuple(
                Card(int(result.principal_variation[index])).code
                for index in range(line_length)
            ),
            contract_succeeded=bool(result.contract_succeeded),
            round_lead=bool(result.round_lead),
            score_after_round=(
                int(result.score_after_round[0]),
                int(result.score_after_round[1]),
            ),
            match_winner=match_winner,
            completion_status=completion,
        )

    @staticmethod
    def _decode_rollout_result(result: _RolloutResultV3) -> NativeRolloutSummary:
        completion = {
            0: "exact",
            1: "budget_exhausted",
            2: "cancelled",
            3: "failed",
            4: "heuristic_complete",
        }.get(int(result.status))
        if completion is None:
            raise NativeUnavailable(
                f"native rollout returned an unknown status: {result.status}"
            )
        match_winner = (
            None if int(result.match_winner) < 0 else Team(int(result.match_winner))
        )
        return NativeRolloutSummary(
            value=float(result.value),
            own_points=float(result.own_points),
            opponent_points=float(result.opponent_points),
            exact=bool(result.exact),
            nodes=int(result.nodes),
            contract_succeeded=bool(result.contract_succeeded),
            round_lead=bool(result.round_lead),
            score_after_round=(
                int(result.score_after_round[0]),
                int(result.score_after_round[1]),
            ),
            match_winner=match_winner,
            completion_status=completion,
        )

    def heuristic_rollout_batch_compact(
        self,
        states: Sequence[RoundState],
        cards: Sequence[Card | int | None],
        team: Team,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[NativeRolloutSummary, ...]:
        """Evaluate paired rollouts without materializing principal variations."""
        if not self.supports_compact_rollout or self.rules_v3 is None:
            raise NativeUnavailable("compact batched rollouts require native ABI V3")
        if len(states) != len(cards):
            raise ValueError("native rollout states and actions must align")
        if not states:
            return ()
        tasks = (_RolloutTaskV3 * len(states))()
        for index, (state, card) in enumerate(zip(states, cards, strict=True)):
            tasks[index].state = self._state_v3(state)
            tasks[index].initial_card = (
                _ROLLOUT_NO_INITIAL_CARD_V3 if card is None else Card.parse(card).id
            )
        results = (_RolloutResultV3 * len(states))()
        cancelled, stop_polling, poller = self._start_cancellation_poller(cancel_check)
        try:
            status = int(
                self.lib.baloot_heuristic_rollout_batch_compact_v3(
                    ctypes.byref(self.rules_v3),
                    tasks,
                    len(states),
                    int(Team(team)),
                    ctypes.byref(cancelled),
                    results,
                )
            )
        finally:
            self._stop_cancellation_poller(stop_polling, poller)
        if status == -2:
            raise InterruptedError("native rollout batch cancelled")
        if status != len(states):
            raise NativeUnavailable(f"native compact rollout batch failed: {status}")
        return tuple(
            self._decode_rollout_result(results[index]) for index in range(len(states))
        )

    def heuristic_rollout_batch(
        self,
        states: Sequence[RoundState],
        cards: Sequence[Card | int | None],
        team: Team,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[NativeExactSearchResult, ...]:
        """Evaluate paired rollouts, optionally from each state's current position."""
        if not self.supports_v3 or self.rules_v3 is None:
            raise NativeUnavailable("batched rollouts require native ABI V3")
        if len(states) != len(cards):
            raise ValueError("native rollout states and actions must align")
        if not states:
            return ()
        tasks = (_RolloutTaskV3 * len(states))()
        for index, (state, card) in enumerate(zip(states, cards, strict=True)):
            tasks[index].state = self._state_v3(state)
            tasks[index].initial_card = (
                _ROLLOUT_NO_INITIAL_CARD_V3 if card is None else Card.parse(card).id
            )
        results = (_SearchResultV3 * len(states))()
        cancelled, stop_polling, poller = self._start_cancellation_poller(cancel_check)
        try:
            status = int(
                self.lib.baloot_heuristic_rollout_batch_v3(
                    ctypes.byref(self.rules_v3),
                    tasks,
                    len(states),
                    int(Team(team)),
                    ctypes.byref(cancelled),
                    results,
                )
            )
        finally:
            self._stop_cancellation_poller(stop_polling, poller)
        if status == -2:
            raise InterruptedError("native rollout batch cancelled")
        if status != len(states):
            raise NativeUnavailable(f"native rollout batch failed: {status}")
        return tuple(self._decode_search_result(results[index]) for index in range(len(states)))

    def exact_team_value(
        self,
        state: RoundState,
        _engine: object,
        team: Team,
        *,
        max_nodes: int = 250_000,
        max_seconds: float = 2.0,
        max_depth: int | None = None,
        max_cache_entries: int = 100_000,
        cancel_check: Callable[[], bool] | None = None,
    ) -> NativeExactSearchResult:
        """Solve one late-game state in one coarse native call.

        V3 is deliberately exact-search only. It refuses early states whose
        project scoring is not yet frozen instead of inventing a second rules
        interpretation.
        """
        if not self.supports_v3 or self.rules_v3 is None:
            raise NativeUnavailable("exact search requires native ABI V3")
        if max_nodes <= 0 or max_seconds <= 0 or max_cache_entries < 0:
            raise ValueError("native exact-search budgets are invalid")
        if max_depth is not None and not 1 <= int(max_depth) <= 32:
            raise ValueError("native exact-search depth must be in [1, 32]")
        native_state = self._state_v3(state)
        limits = _SearchLimitsV3()
        limits.maximum_nodes = int(max_nodes)
        limits.maximum_milliseconds = max(1, int(math.ceil(float(max_seconds) * 1000.0)))
        limits.maximum_cache_entries = int(max_cache_entries)
        limits.maximum_depth = 0 if max_depth is None else int(max_depth)
        limits.team = int(Team(team))
        cancelled, stop_polling, poller = self._start_cancellation_poller(cancel_check)
        result = _SearchResultV3()
        try:
            status = int(
                self.lib.baloot_exact_search_v3(
                    ctypes.byref(self.rules_v3),
                    ctypes.byref(native_state),
                    ctypes.byref(limits),
                    ctypes.byref(cancelled),
                    ctypes.byref(result),
                )
            )
        finally:
            self._stop_cancellation_poller(stop_polling, poller)
        if status != 0:
            raise NativeUnavailable(f"native exact-search request failed: {status}")
        return self._decode_search_result(result)

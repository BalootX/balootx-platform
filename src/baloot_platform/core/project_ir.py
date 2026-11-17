from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Mapping, Sequence, TypeVar

from .cards import FULL_DECK, Card
from .enums import Contract, Rank, Seat, Suit, Team
from .models import Project


ProjectLike = TypeVar("ProjectLike")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def project_id_v1(project: Any) -> str:
    """Return the stable identity shared by project actions and CP-SAT."""

    digest = _canonical_sha256(
        {
            "owner": Seat.parse(project.owner).name.lower(),
            "kind": str(project.kind),
            "cards": [Card.parse(card).id for card in project.cards],
            "value": int(project.value),
            "highRank": int(project.high_rank),
            "categoryPriority": int(project.category_priority),
        }
    )
    return f"PROJECT-{digest[:16].upper()}"


@dataclass(frozen=True, slots=True)
class ProjectDeclarationActionV1:
    actor: Seat
    action: str
    project_id: str | None = None

    @classmethod
    def build(
        cls,
        value: "ProjectDeclarationActionV1 | Mapping[str, Any] | str",
        *,
        actor: Seat | str | int | None = None,
    ) -> "ProjectDeclarationActionV1":
        if isinstance(value, cls):
            if actor is not None and value.actor != Seat.parse(actor):
                raise ValueError("project declaration action actor mismatch")
            return value
        if isinstance(value, str):
            if actor is None:
                raise ValueError("project declaration action requires actor")
            action, separator, project_id = value.partition(":")
            raw: Mapping[str, Any] = {
                "actor": Seat.parse(actor).name.lower(),
                "action": action,
                **({"projectId": project_id} if separator else {}),
            }
        elif isinstance(value, Mapping):
            raw = value
        else:
            raise TypeError("project declaration action must be a mapping or code")
        unknown = set(raw) - {"actor", "action", "projectId"}
        if unknown:
            raise ValueError(
                f"project declaration action contains unknown fields: {sorted(unknown)}"
            )
        parsed_actor = Seat.parse(raw.get("actor") if actor is None else actor)
        if raw.get("actor") is not None and Seat.parse(raw["actor"]) != parsed_actor:
            raise ValueError("project declaration action actor mismatch")
        action = raw.get("action")
        if type(action) is not str or action not in {"pass", "declare"}:
            raise ValueError("project declaration action must be pass or declare")
        project_id = raw.get("projectId")
        if action == "pass":
            if project_id is not None:
                raise ValueError("project pass action cannot contain projectId")
            return cls(parsed_actor, action)
        if (
            type(project_id) is not str
            or not project_id.startswith("PROJECT-")
            or len(project_id) != 24
            or any(character not in "0123456789ABCDEF" for character in project_id[8:])
        ):
            raise ValueError("project declare action requires a stable projectId")
        return cls(parsed_actor, action, project_id)

    @property
    def code(self) -> str:
        return self.action if self.project_id is None else f"declare:{self.project_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.name.lower(),
            "action": self.action,
            **({"projectId": self.project_id} if self.project_id is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class ProjectDeclarationStateV1:
    projects: tuple[Any, ...]
    seats: tuple[Seat, ...]
    next_index: int = 0
    actions: tuple[ProjectDeclarationActionV1, ...] = ()

    @property
    def current_actor(self) -> Seat | None:
        return self.seats[self.next_index] if self.next_index < len(self.seats) else None

    @property
    def complete(self) -> bool:
        return self.next_index == len(self.seats)


@dataclass(frozen=True, slots=True)
class ProjectDomainIRV1:
    """Validated project semantics shared by runtime, reference, and CP-SAT."""

    enabled: bool
    declaration_timing: str
    declaration_mode: str
    declaration_mode_authored: bool
    comparison_timing: str
    reveal_timing: str
    maximum_per_player: int
    cards_may_overlap: bool
    only_winning_team_scores: bool
    sequence_enabled: bool
    sequence_minimum_length: int
    sequence_strength: tuple[int, ...]
    sequence_values: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    four_kind_enabled: bool
    four_kind_eligible_ranks: tuple[Rank, ...]
    four_kind_values: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    baloot_enabled: bool
    baloot_only_hokum: bool
    baloot_value: int
    baloot_declare_on_second_card: bool
    baloot_siri_or_fifty_inclusion: str
    baloot_hundred_inclusion: str
    baloot_single_card_policy: str
    category_priorities: Mapping[str, int]
    equal_hokum_sequence_trump_priority: bool
    sha256: str
    reveal_scope: str = "scoring_team"
    reveal_scope_authored: bool = False
    tie_break_order: str = "declaration"
    tie_break_order_authored: bool = False

    @classmethod
    def from_rules(cls, rules: Mapping[str, Any]) -> "ProjectDomainIRV1":
        try:
            projects = dict(rules["projects"])
            sequences = dict(projects["sequences"])
            four_kind = dict(projects["four_of_kind"])
            baloot = dict(projects["baloot"])
            boolean_values = (
                projects["enabled"],
                projects["cards_may_overlap"],
                projects["only_winning_team_scores"],
                sequences["enabled"],
                four_kind["enabled"],
                baloot["enabled"],
                baloot["only_hokum"],
                baloot["declare_on_second_card"],
            )
            if any(type(value) is not bool for value in boolean_values):
                raise TypeError("project switches must be booleans")
            if type(projects["max_scoring_projects_per_player"]) is not int:
                raise TypeError("project maximum must be an integer")
            if type(sequences["minimum_length"]) is not int:
                raise TypeError("sequence minimum must be an integer")
            if not isinstance(four_kind["eligible_ranks"], (list, tuple)):
                raise TypeError("eligible four-kind ranks must be an array")
            sequence_strength = tuple(
                int(value) for value in rules["card_rules"]["sequence_strength"]
            )
            sequence_values = cls._normalize_values(
                sequences["values_game_points"],
                required_keys=("three", "four", "five_plus"),
            )
            four_kind_values = cls._normalize_values(
                four_kind["values_game_points"],
                required_keys=("ten", "jack", "queen", "king", "ace"),
            )
            eligible_ranks = tuple(
                Rank.parse(value) for value in four_kind["eligible_ranks"]
            )
            trump_priority = rules["disputed_options"][
                "equal_hokum_sequence_trump_priority"
            ]["selected"]
            category_priorities = {
                str(key): int(value)
                for key, value in projects.get(
                    "comparison",
                    {
                        "category_priorities": {
                            "siri": 0,
                            "fifty": 0,
                            "hundred_sequence": 1,
                            "hundred_four_kind": 2,
                            "four_hundred": 2,
                        }
                    },
                )["category_priorities"].items()
            }
            if type(trump_priority) is not bool:
                raise TypeError("equal Hokum sequence priority must be boolean")
            declaration_mode_authored = "declaration_mode" in projects
            declaration_mode = str(projects.get("declaration_mode", "mandatory"))
            reveal_scope = projects.get("reveal_scope", "scoring_team")
            tie_break_order = projects.get("tie_break_order", "declaration")
            if tie_break_order not in {"declaration", "second_trick_reveal"}:
                raise ValueError("unsupported project tie-break order")
            if reveal_scope not in {"scoring_team", "comparison_winner"}:
                raise ValueError("unsupported project reveal scope")
            if reveal_scope == "comparison_winner" and trump_priority:
                raise ValueError("winner-only disclosure requires suit-independent comparison")
            core = {
                "schemaVersion": 1,
                "enabled": projects["enabled"],
                "declarationTiming": str(projects["declaration_timing"]),
                "comparisonTiming": str(projects["comparison_timing"]),
                "revealTiming": str(projects["reveal_timing"]),
                "maximumPerPlayer": projects["max_scoring_projects_per_player"],
                "cardsMayOverlap": projects["cards_may_overlap"],
                "onlyWinningTeamScores": projects["only_winning_team_scores"],
                "sequenceEnabled": sequences["enabled"],
                "sequenceMinimumLength": sequences["minimum_length"],
                "sequenceStrength": list(sequence_strength),
                "sequenceValues": cls._values_dict(sequence_values),
                "fourKindEnabled": four_kind["enabled"],
                "fourKindEligibleRanks": [rank.name.lower() for rank in eligible_ranks],
                "fourKindValues": cls._values_dict(four_kind_values),
                "balootEnabled": baloot["enabled"],
                "balootOnlyHokum": baloot["only_hokum"],
                "balootValue": int(baloot["value_game_points"]),
                "balootDeclareOnSecondCard": baloot["declare_on_second_card"],
                "balootSiriOrFiftyInclusion": str(baloot["siri_or_fifty_inclusion"]),
                "balootHundredInclusion": str(baloot["hundred_inclusion"]),
                "balootSingleCardPolicy": str(baloot["single_card_in_other_project"]),
                "categoryPriorities": category_priorities,
                "equalHokumSequenceTrumpPriority": trump_priority,
            }
            if declaration_mode_authored:
                core["declarationMode"] = declaration_mode
            if "reveal_scope" in projects:
                core["revealScope"] = reveal_scope
            if "tie_break_order" in projects:
                core["tieBreakOrder"] = tie_break_order
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "incomplete or invalid project domain configuration"
            ) from exc

        if core["declarationTiming"] != "first_trick_turn":
            raise ValueError("only first-trick project declaration is supported")
        if declaration_mode not in {"mandatory", "optional"}:
            raise ValueError("project declaration mode is unsupported")
        if core["comparisonTiming"] != "after_first_trick":
            raise ValueError("only after-first-trick project comparison is supported")
        if core["revealTiming"] != "during_second_trick":
            raise ValueError("only second-trick project reveal is supported")
        if core["maximumPerPlayer"] <= 0:
            raise ValueError("project maximum per player must be positive")
        if not core["onlyWinningTeamScores"]:
            raise ValueError("only winning-team project scoring is supported")
        if core["sequenceMinimumLength"] not in {3, 4, 5}:
            raise ValueError("sequence minimum length must be three, four, or five")
        if len(sequence_strength) != 8 or set(sequence_strength) != set(range(8)):
            raise ValueError(
                "sequence strength must be a permutation of zero through seven"
            )
        if not eligible_ranks or len(eligible_ranks) != len(set(eligible_ranks)):
            raise ValueError("eligible four-kind ranks must be unique and non-empty")
        if any(
            rank not in {Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE}
            for rank in eligible_ranks
        ):
            raise ValueError("unsupported four-kind rank")
        if not core["balootOnlyHokum"]:
            raise ValueError("Baloot is supported only in Hokum")
        if not core["balootDeclareOnSecondCard"]:
            raise ValueError("Baloot must be declared on the second card")
        if core["balootSiriOrFiftyInclusion"] != "automatic":
            raise ValueError("Siri/Fifty Baloot inclusion must be automatic")
        if core["balootHundredInclusion"] not in {"automatic", "blocked"}:
            raise ValueError("unsupported Hundred Baloot inclusion policy")
        if core["balootSingleCardPolicy"] != "blocked":
            raise ValueError("single-card Baloot overlap must be blocked")
        if (
            set(category_priorities)
            != {
                "siri",
                "fifty",
                "hundred_sequence",
                "hundred_four_kind",
                "four_hundred",
            }
            or min(category_priorities.values()) < 0
        ):
            raise ValueError("project category priorities are incomplete or invalid")
        if core["balootValue"] < 0:
            raise ValueError("Baloot value must be non-negative")

        return cls(
            core["enabled"],
            core["declarationTiming"],
            declaration_mode,
            declaration_mode_authored,
            core["comparisonTiming"],
            core["revealTiming"],
            core["maximumPerPlayer"],
            core["cardsMayOverlap"],
            core["onlyWinningTeamScores"],
            core["sequenceEnabled"],
            core["sequenceMinimumLength"],
            sequence_strength,
            sequence_values,
            core["fourKindEnabled"],
            eligible_ranks,
            four_kind_values,
            core["balootEnabled"],
            core["balootOnlyHokum"],
            core["balootValue"],
            core["balootDeclareOnSecondCard"],
            core["balootSiriOrFiftyInclusion"],
            core["balootHundredInclusion"],
            core["balootSingleCardPolicy"],
            category_priorities,
            core["equalHokumSequenceTrumpPriority"],
            _canonical_sha256(core),
            reveal_scope,
            "reveal_scope" in projects,
            tie_break_order,
            "tie_break_order" in projects,
        )

    @staticmethod
    def _normalize_values(
        source: Mapping[str, Any], *, required_keys: tuple[str, ...]
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        if set(source) != {"sun", "hokum"}:
            raise ValueError("project values require exact Sun and Hokum tables")
        rows = []
        for contract in ("sun", "hokum"):
            table = dict(source[contract])
            if set(table) != set(required_keys):
                raise ValueError("project value table has missing or unknown keys")
            values = tuple((key, int(table[key])) for key in required_keys)
            if any(value < 0 for _key, value in values):
                raise ValueError("project values must be non-negative")
            rows.append((contract, values))
        return tuple(rows)

    @staticmethod
    def _values_dict(
        values: tuple[tuple[str, tuple[tuple[str, int], ...]], ...],
    ) -> dict[str, dict[str, int]]:
        return {contract: dict(table) for contract, table in values}

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schemaVersion": 1,
            "enabled": self.enabled,
            "declarationTiming": self.declaration_timing,
            "comparisonTiming": self.comparison_timing,
            "revealTiming": self.reveal_timing,
            "maximumPerPlayer": self.maximum_per_player,
            "cardsMayOverlap": self.cards_may_overlap,
            "onlyWinningTeamScores": self.only_winning_team_scores,
            "sequenceEnabled": self.sequence_enabled,
            "sequenceMinimumLength": self.sequence_minimum_length,
            "sequenceStrength": list(self.sequence_strength),
            "sequenceValues": self._values_dict(self.sequence_values),
            "fourKindEnabled": self.four_kind_enabled,
            "fourKindEligibleRanks": [
                rank.name.lower() for rank in self.four_kind_eligible_ranks
            ],
            "fourKindValues": self._values_dict(self.four_kind_values),
            "balootEnabled": self.baloot_enabled,
            "balootOnlyHokum": self.baloot_only_hokum,
            "balootValue": self.baloot_value,
            "balootDeclareOnSecondCard": self.baloot_declare_on_second_card,
            "balootSiriOrFiftyInclusion": self.baloot_siri_or_fifty_inclusion,
            "balootHundredInclusion": self.baloot_hundred_inclusion,
            "balootSingleCardPolicy": self.baloot_single_card_policy,
            "categoryPriorities": dict(self.category_priorities),
            "equalHokumSequenceTrumpPriority": (
                self.equal_hokum_sequence_trump_priority
            ),
            "sha256": self.sha256,
        }
        if self.declaration_mode_authored:
            result["declarationMode"] = self.declaration_mode
        if self.reveal_scope_authored:
            result["revealScope"] = self.reveal_scope
        if self.tie_break_order_authored:
            result["tieBreakOrder"] = self.tie_break_order
        return result

    def sequence_strength_for(self, card: Card) -> int:
        return self.sequence_strength[int(Card.parse(card).rank)]

    def sequence_value_for(self, contract: Contract, key: str) -> int:
        table = self._values_dict(self.sequence_values)[
            Contract.parse(contract).effective.value
        ]
        try:
            return int(table[str(key)])
        except KeyError as exc:
            raise ValueError(f"unsupported sequence project value key: {key}") from exc

    def four_kind_value_for(self, contract: Contract, rank: Rank | str | int) -> int:
        parsed_rank = Rank.parse(rank)
        if parsed_rank not in self.four_kind_eligible_ranks:
            return 0
        table = self._values_dict(self.four_kind_values)[
            Contract.parse(contract).effective.value
        ]
        return int(table[parsed_rank.name.lower()])

    def should_announce(self, trick_index: int) -> bool:
        return self.declaration_timing == "first_trick_turn" and trick_index == 0

    def should_compare(self, completed_trick_index: int) -> bool:
        return (
            self.comparison_timing == "after_first_trick" and completed_trick_index == 0
        )

    def should_reveal(self, trick_index: int) -> bool:
        return self.reveal_timing == "during_second_trick" and trick_index == 1

    def _sequence_value(self, length: int, contract: Contract) -> tuple[str, int, int]:
        table = self._values_dict(self.sequence_values)[
            Contract.parse(contract).effective.value
        ]
        if length >= 5:
            return (
                "hundred_sequence",
                table["five_plus"],
                self.category_priorities["hundred_sequence"],
            )
        if length == 4:
            return "fifty", table["four"], self.category_priorities["fifty"]
        return "siri", table["three"], self.category_priorities["siri"]

    def _four_kind_value(self, rank: Rank, contract: Contract) -> tuple[str, int, int]:
        effective = Contract.parse(contract).effective
        value = self._values_dict(self.four_kind_values)[effective.value][
            rank.name.lower()
        ]
        kind = (
            "four_hundred"
            if effective == Contract.SUN and rank == Rank.ACE
            else "hundred_four_kind"
        )
        return kind, value, self.category_priorities[kind]

    def candidates(
        self, owner: Seat, cards: Sequence[Card], contract: Contract
    ) -> tuple[Project, ...]:
        if not self.enabled:
            return ()
        owner = Seat.parse(owner)
        hand = tuple(Card.parse(card) for card in cards)
        contract = Contract.parse(contract)
        result: list[Project] = []
        if self.sequence_enabled:
            for suit in Suit:
                suited = sorted(
                    (card for card in hand if card.suit == suit),
                    key=self.sequence_strength_for,
                )
                by_strength = {
                    self.sequence_strength_for(card): card for card in suited
                }
                runs: list[list[int]] = []
                current: list[int] = []
                for strength in sorted(by_strength):
                    if not current or strength == current[-1] + 1:
                        current.append(strength)
                    else:
                        if len(current) >= self.sequence_minimum_length:
                            runs.append(current)
                        current = [strength]
                if len(current) >= self.sequence_minimum_length:
                    runs.append(current)
                for run in runs:
                    for length in range(
                        self.sequence_minimum_length, min(5, len(run)) + 1
                    ):
                        for start in range(0, len(run) - length + 1):
                            window = run[start : start + length]
                            selected = tuple(by_strength[value] for value in window)
                            kind, value, priority = self._sequence_value(
                                length, contract
                            )
                            result.append(
                                Project(
                                    owner,
                                    kind,
                                    selected,
                                    value,
                                    max(window),
                                    priority,
                                )
                            )
        if self.four_kind_enabled:
            for rank in sorted(self.four_kind_eligible_ranks, key=int):
                group = tuple(Card.of(suit, rank) for suit in Suit)
                if all(card in hand for card in group):
                    kind, value, priority = self._four_kind_value(rank, contract)
                    result.append(
                        Project(owner, kind, group, value, int(rank), priority)
                    )
        return tuple(result)

    @staticmethod
    def group_score(group: Sequence[Any]) -> tuple[int, int, int, int]:
        if not group:
            return 0, 0, 0, 0
        return (
            sum(int(project.value) for project in group),
            max(int(project.category_priority) for project in group),
            max(int(project.high_rank) for project in group),
            -sum(len(project.cards) for project in group),
        )

    def group_is_valid(self, group: Sequence[Any]) -> bool:
        if len(group) > self.maximum_per_player:
            return False
        mask = 0
        for project in group:
            if not self.cards_may_overlap and mask & int(project.card_mask):
                return False
            mask |= int(project.card_mask)
        return True

    def select(self, candidates: Sequence[ProjectLike]) -> tuple[ProjectLike, ...]:
        best: tuple[tuple[int, int, int, int], tuple[ProjectLike, ...]] = (
            (0, 0, 0, 0),
            (),
        )
        for size in range(1, min(self.maximum_per_player, len(candidates)) + 1):
            for group in combinations(candidates, size):
                if not self.group_is_valid(group):
                    continue
                score = self.group_score(group)
                if score > best[0]:
                    best = (score, group)
        return best[1]

    def rank_key(
        self,
        project: Any,
        *,
        trump_suit: Suit | None = None,
    ) -> tuple[int, int, int, int]:
        in_trump = int(
            self.equal_hokum_sequence_trump_priority
            and trump_suit is not None
            and project.cards
            and all(card.suit == trump_suit for card in project.cards)
            and project.kind in {"siri", "fifty", "hundred_sequence"}
        )
        return (
            int(project.value),
            int(project.category_priority),
            int(project.high_rank),
            in_trump,
        )

    def detect(
        self,
        hands: Mapping[Seat, Sequence[Card]],
        contract: Contract,
        dealer: Seat,
    ) -> list[Project]:
        if not self.enabled:
            return []
        result: list[Project] = []
        for seat in Seat:
            result.extend(self.select(self.candidates(seat, hands[seat], contract)))
        order = self.declaration_seats(dealer)
        result.sort(
            key=lambda project: (
                order.index(project.owner),
                -project.value,
                -project.category_priority,
                -project.high_rank,
            )
        )
        return result

    @staticmethod
    def declaration_seats(dealer: Seat) -> tuple[Seat, ...]:
        dealer = Seat.parse(dealer)
        first = dealer.next()
        return (first, first.next(), first.next().next(), dealer)

    def declaration_state(
        self,
        projects: Sequence[ProjectLike],
        *,
        dealer: Seat,
    ) -> ProjectDeclarationStateV1:
        if any(bool(project.declared) for project in projects):
            raise ValueError("project declaration state must start before declarations")
        return ProjectDeclarationStateV1(
            tuple(projects),
            self.declaration_seats(dealer),
        )

    def _declarable_for_turn(
        self,
        projects: Sequence[ProjectLike],
        seat: Seat,
        *,
        trump_suit: Suit | None,
    ) -> ProjectLike | None:
        candidate = max(
            (project for project in projects if project.owner == seat),
            default=None,
            key=lambda project: (
                project.value,
                project.category_priority,
                project.high_rank,
            ),
        )
        previous = max(
            (project for project in projects if project.declared),
            default=None,
            key=lambda project: (
                project.value,
                project.category_priority,
                project.high_rank,
                -project.declaration_order,
            ),
        )
        if candidate is None:
            return None
        if previous is not None and self.rank_key(
            candidate,
            trump_suit=trump_suit,
        ) < self.rank_key(
            previous,
            trump_suit=trump_suit,
        ):
            return None
        return candidate

    def legal_declaration_actions(
        self,
        state: ProjectDeclarationStateV1,
        *,
        trump_suit: Suit | None,
    ) -> tuple[ProjectDeclarationActionV1, ...]:
        if not isinstance(state, ProjectDeclarationStateV1):
            raise TypeError("project declaration actions require ProjectDeclarationStateV1")
        actor = state.current_actor
        if actor is None:
            return ()
        candidate = (
            self._declarable_for_turn(
                state.projects,
                actor,
                trump_suit=trump_suit,
            )
            if self.enabled
            else None
        )
        pass_action = ProjectDeclarationActionV1(actor, "pass")
        if candidate is None:
            return (pass_action,)
        declare_action = ProjectDeclarationActionV1(
            actor,
            "declare",
            project_id_v1(candidate),
        )
        return (
            (pass_action, declare_action)
            if self.declaration_mode == "optional"
            else (declare_action,)
        )

    def apply_declaration_action(
        self,
        state: ProjectDeclarationStateV1,
        action: ProjectDeclarationActionV1 | Mapping[str, Any] | str,
        *,
        trump_suit: Suit | None,
    ) -> tuple[ProjectDeclarationStateV1, ProjectLike | None]:
        actor = state.current_actor
        if actor is None:
            raise ValueError("project declaration sequence is already complete")
        parsed = ProjectDeclarationActionV1.build(action, actor=actor)
        legal = self.legal_declaration_actions(state, trump_suit=trump_suit)
        if parsed.code not in {item.code for item in legal}:
            raise ValueError("project declaration action is illegal for the current turn")
        projects = list(state.projects)
        announced = None
        if parsed.project_id is not None:
            announced = next(
                (
                    project
                    for project in projects
                    if project.owner == actor and project_id_v1(project) == parsed.project_id
                ),
                None,
            )
            if announced is None:
                raise ValueError("project declaration action references an unknown project")
            order = 1 + sum(1 for project in projects if project.declared)
            projects = [
                replace(project, declared=True, declaration_order=order)
                if project is announced
                else project
                for project in projects
            ]
            announced = replace(announced, declared=True, declaration_order=order)
        return (
            ProjectDeclarationStateV1(
                tuple(projects),
                state.seats,
                state.next_index + 1,
                (*state.actions, parsed),
            ),
            announced,
        )

    def announce_best(
        self, projects: Sequence[ProjectLike], seat: Seat
    ) -> tuple[list[ProjectLike], ProjectLike | None]:
        owned = [project for project in projects if project.owner == seat]
        if not owned:
            return list(projects), None
        chosen = max(
            owned,
            key=lambda project: (
                project.value,
                project.category_priority,
                project.high_rank,
            ),
        )
        order = 1 + sum(1 for project in projects if project.declared)
        updated = [
            replace(project, declared=True, declaration_order=order)
            if project == chosen
            else project
            for project in projects
        ]
        return updated, replace(chosen, declared=True, declaration_order=order)

    def announce_for_turn(
        self,
        projects: Sequence[ProjectLike],
        seat: Seat,
        *,
        trump_suit: Suit | None,
    ) -> tuple[list[ProjectLike], ProjectLike | None]:
        if self.declaration_mode != "mandatory":
            raise ValueError(
                "optional project declarations require an explicit declaration action"
            )
        if any(project.owner == seat and project.declared for project in projects):
            return list(projects), None
        candidate = self._declarable_for_turn(
            projects,
            seat,
            trump_suit=trump_suit,
        )
        if candidate is None:
            return list(projects), None
        return self.announce_best(projects, seat)

    def resolve_declarations(
        self,
        projects: Sequence[ProjectLike],
        *,
        dealer: Seat,
        trump_suit: Suit | None,
        second_trick_leader: Seat | None = None,
    ) -> tuple[list[ProjectLike], Team | None]:
        resolved = list(projects)
        for seat in self.declaration_seats(dealer):
            resolved, _announced = self.announce_for_turn(
                resolved,
                seat,
                trump_suit=trump_suit,
            )
        return self.score_team(
            resolved,
            trump_suit=trump_suit,
            second_trick_leader=second_trick_leader,
        )

    def score_team(
        self,
        projects: Sequence[ProjectLike],
        *,
        trump_suit: Suit | None = None,
        second_trick_leader: Seat | None = None,
    ) -> tuple[list[ProjectLike], Team | None]:
        best = self.comparison_winner(projects, trump_suit=trump_suit,
                                      second_trick_leader=second_trick_leader)
        if best is None:
            return list(projects), None
        winning = best.team
        return (
            [replace(project, scoring=project.team == winning) for project in projects],
            winning,
        )

    def comparison_winner(
        self,
        projects: Sequence[ProjectLike],
        *,
        trump_suit: Suit | None = None,
        second_trick_leader: Seat | None = None,
    ) -> ProjectLike | None:
        """Return the one declared project that wins authoritative comparison."""

        if self.tie_break_order == "second_trick_reveal":
            if second_trick_leader is None:
                raise ValueError("project comparison requires the second-trick leader")
            second_trick_leader = Seat.parse(second_trick_leader)
        return max(
            (project for project in projects if project.declared),
            default=None,
            key=lambda project: (
                *self.rank_key(
                    project,
                    trump_suit=trump_suit,
                ),
                (int(project.team == second_trick_leader.team),
                 -((int(project.owner) - int(second_trick_leader)) % 4))
                if self.tie_break_order == "second_trick_reveal"
                else -project.declaration_order,
            ),
        )

    @staticmethod
    def project_points(projects: Sequence[Any]) -> tuple[int, int]:
        points = [0, 0]
        for project in projects:
            if project.scoring:
                points[int(project.team)] += int(project.value)
        return points[0], points[1]

    def baloot_project_state(
        self,
        projects: Sequence[Any],
        *,
        owner: Seat,
        trump_suit: Suit,
    ) -> tuple[bool, bool]:
        pair = {
            Card.of(trump_suit, Rank.KING),
            Card.of(trump_suit, Rank.QUEEN),
        }
        included = False
        blocked = False
        for project in projects:
            if project.owner != owner or not bool(project.scoring):
                continue
            project_cards = set(project.cards)
            if (
                self.baloot_siri_or_fifty_inclusion == "automatic"
                and project.kind in {"siri", "fifty"}
                and pair.issubset(project_cards)
            ):
                included = True
            if (
                self.baloot_hundred_inclusion in {"automatic", "blocked"}
                and project.kind == "hundred_sequence"
                and pair.issubset(project_cards)
            ):
                if self.baloot_hundred_inclusion == "automatic":
                    included = True
                else:
                    blocked = True
            if (
                self.baloot_hundred_inclusion == "blocked"
                and project.kind == "hundred_four_kind"
                and project.cards
                and project.cards[0].rank in {Rank.KING, Rank.QUEEN}
            ):
                blocked = True
            if (
                self.baloot_single_card_policy == "blocked"
                and len(pair & project_cards) == 1
            ):
                blocked = True
        return included, blocked

    def baloot_eligible_seats(
        self,
        hands: Mapping[Seat, Sequence[Card]],
        projects: Sequence[Any],
        contract: Contract,
        trump_suit: Suit | None,
    ) -> frozenset[Seat]:
        if (
            not self.baloot_enabled
            or (self.baloot_only_hokum and Contract.parse(contract) != Contract.HOKUM)
            or trump_suit is None
        ):
            return frozenset()
        pair = {
            Card.of(trump_suit, Rank.KING),
            Card.of(trump_suit, Rank.QUEEN),
        }
        seats: set[Seat] = set()
        for seat in Seat:
            if not pair.issubset(set(hands[seat])):
                continue
            included, blocked = self.baloot_project_state(
                projects,
                owner=seat,
                trump_suit=trump_suit,
            )
            if included or not blocked:
                seats.add(seat)
        return frozenset(seats)

    def complete_baloot_teams(
        self,
        hands: Mapping[Seat, Sequence[Card]],
        projects: Sequence[Any],
        contract: Contract,
        trump_suit: Suit | None,
    ) -> frozenset[Team]:
        return frozenset(
            seat.team
            for seat in self.baloot_eligible_seats(
                hands,
                projects,
                contract,
                trump_suit,
            )
        )

    def reveal_for_turn(
        self, projects: Sequence[ProjectLike], seat: Seat, *,
        trump_suit: Suit | None = None, second_trick_leader: Seat | None = None,
    ) -> tuple[list[ProjectLike], tuple[ProjectLike, ...]]:
        winner = (self.comparison_winner(projects, trump_suit=trump_suit,
            second_trick_leader=second_trick_leader)
            if self.reveal_scope == "comparison_winner" else None)
        revealed = tuple(
            project
            for project in projects
            if project.owner == seat and project.scoring and not project.revealed
            and (self.reveal_scope != "comparison_winner" or project is winner)
        )
        if not revealed:
            return list(projects), ()
        updated = [
            replace(project, revealed=True)
            if any(project is item for item in revealed)
            else project
            for project in projects
        ]
        return updated, tuple(replace(project, revealed=True) for project in revealed)

    def card_patterns(
        self, *, kind: str, value: int, high_rank: int | None, contract: Contract
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    project.card_mask
                    for project in self.candidates(Seat.NORTH, FULL_DECK, contract)
                    if project.kind == kind
                    and project.value == int(value)
                    and (high_rank is None or project.high_rank == int(high_rank))
                }
            )
        )


__all__ = [
    "ProjectDeclarationActionV1",
    "ProjectDeclarationStateV1",
    "ProjectDomainIRV1",
    "project_id_v1",
]

# Research scenario V4

Research requests combine an exact input or generated search with typed constraints,
a decision point, an outcome predicate, and a frozen proof budget. The public
contracts live in `src/baloot_platform/research/scenario_contracts.py`.

## Predicate capabilities

`ScenarioPredicateRegistryV1.catalog()` defines each predicate's arguments,
stage, allowed hardness, construction capability, verifier, and information
boundary. A postcondition must be verified after construction; it must not be
presented as a direct construction capability. Report-only observations must not
be accepted as hard constraints. Inactive or unsupported predicates fail closed.
The executable inventory is in `tests/test_research_scenario_predicate_inventory_v1.py`.

## Rules and proofs

Rules and bundle hashes bind a request to its configuration authority. The
[generated engine guide](ENGINE_GUIDE_AR.md) explains that public rule profile.
A replay-valid trace, a bounded proof, and an exact proof are distinct results.
Requested horizons and declared resource limits must accompany any quality claim.
An incomplete or exhausted search does not prove infeasibility or optimality.

## Observer and publication boundaries

Each perspective receives its own public observation and authorized private hand.
Other hands, future plays, hidden solver state, and unrevealed terminal scores must
not enter observer-facing explanations or media. Referee-only evidence follows the
[adjudication boundary](ADJUDICATION_RESEARCH_BOUNDARY.md).

Rendering and publication retain their own truth and compatibility checks. A unit
fixture or local benchmark does not establish external certification, production
qualification, or deployment authorization.

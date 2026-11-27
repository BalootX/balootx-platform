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

### ENGINE-INTEREST v1

Posting requires complete viewer-world coverage and a verified viewer-seat exact
proof. The scorer requires a lowercase SHA-256 digest of that proof receipt and
records it as `exactActorProofSha256` in the scorecard.

Pass the full confirmed history for the target platform, newest first. Use a
stable lowercase SHA-256 `dealHash` across every decision point and render of a
deal. A repeated
`exactStateId`, deal hash, visible opener, or caption blocks admission anywhere in
that history. The family and mechanism gates compare against the latest confirmed
post. A family match anywhere in the last 14 posts gives zero variety points.
The score floor and ranking use the full-precision sum of the score parts. The
one-decimal score is for display.

### Nightly synthetic pool

The seal considers qualified shard candidates by descending full-precision
ENGINE-INTEREST score, with ties resolved by ascending candidate ID. It excludes
any candidate that shares an `exactStateId`, `dealHash`, `scenarioFamily` or
`visibleOpenerSha256` with an admitted candidate. The seal reports the first
matching key in that order.

The `NEVER_POST` pool requires at least 14 admitted candidates across at least
three mechanisms. The seal logs the counts and repeat exclusions even if a
minimum fails. On success, `candidate-index.json` lists only admitted candidates
under `candidates` and `qualified`. Its `exclusions` entries record the excluded
`id`, a `repeat_<key>` reason and the admitted `repeatsId`. Only admitted media is
copied into the sealed pool.

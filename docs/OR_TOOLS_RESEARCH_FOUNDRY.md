# OR-Tools research foundry

The foundry separates candidate construction, legal replay, proof, and publication.
A generated candidate or a completed solver call does not by itself qualify a
result. Requests freeze their configuration authority, scenario identity, and
resource budgets before execution.

## Rule authority and compiler coverage

Public bundle profiles under `configs/bundle_ortools_research_*.yaml` select the
rules, conventions, challenge policy, and product vocabulary. Rule source catalogs
bind each primitive to its retained source, runtime owner, CP-SAT owner, executable
examples, and counterexamples. The generated [engine guide](ENGINE_GUIDE_AR.md)
explains the active rule profile.

`BalootDomainCompilerV1` emits runtime, reference, CP-SAT, policy, documentation,
and behavioral-proof views of those primitives. A supported primitive needs real
coverage and source bindings. Missing coverage must remain unsupported; the
compiler must not substitute a reference result for native CP-SAT qualification.
`tests/test_rule_leaf_mutation_gate.py` exercises the active leaves and rejection
boundaries against ephemeral configuration mutations.

## Execution and evidence

The executor normalizes the exact state, constructs only legal decisions, and
tracks the remaining shared wall, CPU, node, depth, and hidden-world limits.
Budget exhaustion is incomplete evidence, rather than proof that a requested
outcome is impossible. Exact claims require complete declared coverage; bounded
claims retain their horizon and evaluator. [Scenario V4](RESEARCH_SCENARIO_V4.md)
documents the typed request and predicate boundaries.

Observer-facing results contain only the selected observer's authorized view.
Hidden worlds, other players' hands, future actions, and internal proof artifacts
must remain outside that view. Rendering and publishing apply their own frozen
truth and compatibility policies. Synthetic unit fixtures test these contracts
without establishing external certification, retained historical qualification,
or deployment authorization.

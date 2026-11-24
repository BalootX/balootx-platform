# BalootX

## Project facts

- Repo: [BalootX/balootx-platform](https://github.com/BalootX/balootx-platform)
- Main branch: `main`
- Board (private): [BalootX](https://github.com/orgs/BalootX/projects/1)
- Release plan: [milestones page](https://github.com/BalootX/balootx-platform/milestones)
- Leads: BalootX Master, BalootX Delivery, BalootX QA, BalootX Design
- UI project: yes
- Other repos: none
- Environments used: [CI](DELIVERY.md#ci)

## Links

- [Delivery](DELIVERY.md) and [design](DESIGN.md)
- Engineering standard (version in [DELIVERY.md](DELIVERY.md)) and the project-delivery-orchestration skill

## Product and current goal

BalootX combines a Saudi Baloot engine, the Foundry research dashboard, and rendered clips that show card-play decisions. The goal is to turn legally replayable synthetic scenarios into verifiable observer-safe explanations and media.

## Architecture and sources of truth

- The Python engine owns rules, transitions and replay. The [authority map](docs/architecture/authority-map.json) identifies the owner for each contract.
- The [agent profile schema](schemas/profiles/agents.schema.json) validates both bundled and standalone lineups. Row errors give the file, JSON Pointer and correction. The shipped [v1 lineup](configs/agents/adaptive_lineup_v1.yaml) uses its own required fields and derives its search tuning from its budget.
- The [OR-Tools Foundry guide](docs/OR_TOOLS_RESEARCH_FOUNDRY.md) covers rule and proof authority. The [Scenario V4 contract](docs/RESEARCH_SCENARIO_V4.md) covers research requests, result evidence and engine-interest criteria.
- The FastAPI service serves the [Foundry dashboard](src/baloot_platform/service/foundry_dashboard) at `/foundry`. `/studio` redirects there. The [OpenAPI contract](api) drives generated clients.
- The [video package](src/baloot_platform/video) projects the observer's replay into a render bundle for the Remotion compositions. See [the render design](DESIGN.md#rendered-clips) for the visual authorities.

## Commands

Run the Python commands at the repository root and npm commands in the named package. Use the [CI-only PC rule](DELIVERY.md#ci-only-pc-rule) to choose where to run them.

| Task | Command |
| --- | --- |
| Python setup (3.12 or 3.13) | `python -m pip install -e ".[dev,grpc,security]"` |
| Foundry and Remotion setup | `npm ci --ignore-scripts --no-audit --no-fund` |
| Python lint | `python -m ruff check --select E9,F63,F7,F82 src scripts sdk/python tests` |
| Format edited Python files | `python -m ruff format <edited-python-files>` |
| Typecheck | Foundry `npm run check`. Remotion `npm run typecheck`. |
| Fast test | `python -m pytest -q tests/test_ci_hosted_contract.py tests/test_repository_structure.py tests/test_media_render_artifact.py` |
| Full test | Use the hosted full profile in [DELIVERY.md](DELIVERY.md#ci). |
| Build | Python `python -m pip wheel --no-deps .`. Foundry `npm run build`. |
| Run on a development host | `uvicorn baloot_platform.service.api:app --host 127.0.0.1 --port 8080`. Open `/foundry` after building the dashboard. |

## Coding rules

- Before adding code, configuration or tooling, look for something to reuse in the repo, related projects and existing work. Cite the search and current primary documentation in the PR when adding a component, file, dependency or contract. Keep temporary workarounds tied to a dated bug for the root fix. Keep one owner for each rule and value. Use validated, versioned configuration for values that vary by installation and for business rules that can be added as rows. Record who changed configuration, when and why.
- Read callers before editing. Reproduce bugs, keep the PR to one purpose, and update the owning docs and affected tests with the change. Test changed logic with a unit or regression test, exercise external boundaries, and reserve end-to-end tests for core flows. Use a fixed clock, controlled inputs and explicit synchronization.
- Validate API, import and configuration inputs at entry and explain rejected values. Version APIs used by independently deployed clients. Keep client-facing API versions and the previous release's schema working during rollouts. Keep applied migrations immutable. Add a forward migration for schema corrections. Ship destructive cleanup separately after callers have moved.
- Check actor, role and ownership on the server before serving protected data or mutating it. Reject access unless explicitly allowed, and test allowed and denied requests for each affected role.
- Store instants in UTC and local schedules with their named time zone. Give background jobs and message handlers a dedupe key. Set deadlines and bounded backoff for outbound calls. Reconcile an uncertain external write before another attempt.
- Use structured logs with a request or trace ID. Log BalootX IDs for players, reviewers and stored records. Keep names and record contents out of logs. Return safe errors with a next step. Page growing lists and check database query plans for indexing and per-row lookups.
- Use the [design contract](DESIGN.md) for UI changes. Preserve Arabic and right-to-left layout, source visible copy from translations, and verify the changed flows against WCAG 2.2 AA.
- Preserve the public rules and replay contracts at their owning source. Update generated views and packaged copies from that source. Use `python scripts/sync_package_resources.py` after changing mirrored resources and `python scripts/generate_openapi.py --check` to check the OpenAPI output.
- Treat a valid replay, a bounded result and an exact proof as separate evidence. Preserve a result's scope and budget when exposing it through the API, UI or render bundle.

## Critical paths

Apply the [independent review procedure](DELIVERY.md#merge) only when a change affects one of these controls. The [critical path configuration](ci/critical-paths.json) maps them to repository paths. All other changes merge on green required CI.

- **Production** includes deploys, production configuration and production data.
- **Secrets and credentials** include how tokens and keys are read, scoped, stored and rotated, including workflow secrets, token permissions, secret scanning and credential handling.
- **Real people's or customer data** includes migrations, deletion, retention, audit history, backups, restore and access to personal or restricted records. Uncovered safety-critical rules also require review.
- **Making anything public** includes the posting controller, publication ledger writer, caption and media release steps, and any transition from `NEVER_POST` to posting.
- **Hosts, networks and firewalls** include runner and render-host configuration, network access and firewall rules.
- **Repo settings, branch protection and release gates** include rulesets, required checks, the privacy deny check, clip admission checks, and removing, skipping or weakening a test or the test-file guard. The clip release gate's r8 revision requires its one independent review.

Once end-to-end tests cover every BalootX safety rule and this section records each rule and its test, changes to covered rules merge on green CI. A new safety rule needs one review until its test lands.

## Data safety

- Use synthetic fixtures for tests, previews and public examples, and local services for integration tests. Enforce the observer boundary for hands, future actions and solver evidence, and follow the [adjudication boundary](docs/ADJUDICATION_RESEARCH_BOUNDARY.md) for restricted evidence.
- Preserve immutable release evidence. Follow the [rollback procedure](DELIVERY.md#rollback) when a shipped change must be reversed.

The service can hold these records outside Git.

| Personal data | Retention and deletion |
| --- | --- |
| Account and guest IDs, usernames, display names, password hashes and salts, friend links and requests | The service has no automatic expiry or account deletion flow for these rows. |
| Session IDs, token and user-agent hashes, CSRF state, and activity timestamps | Guest and local-account sessions expire according to the [identity profile](configs/identity/optional_simple_v1.yaml). Logout revokes access. The service has no automatic row purge. |
| Linked sign-in provider IDs and claims, which can include name and email, and passkey IDs, public keys, labels and usage metadata | The service has no automatic expiry or unlink/delete flow for these rows. Sign-in transactions, passkey challenges and WebSocket tickets expire or become unusable after consumption. Their backing rows have no automatic purge. |
| Room memberships, player display names and live-room reconnect-token hashes, account-linked game and attempt history, review and adjudication records, consent and other actor-linked audit records | These records have no project-wide retention period or deletion procedure. Keep them out of public fixtures and artifacts. |
| Player IDs, ability estimates and challenge response timings | Estimates are updated in place and responses are stored separately. The service has no automatic expiry or deletion flow. PostgreSQL deletes responses when their player or challenge is deleted. |
| Reviewer IDs, role verification and reputation profiles | The current profile is updated in place. The service has no automatic expiry or deletion flow. |
| Free-text decision feedback and optional reviewer names | The service has no automatic expiry or deletion flow. PostgreSQL deletes feedback when its decision is deleted. |
| Request network addresses and rate-limit activity | When enabled, the limiter keeps client network addresses as keys in process memory until restart. A later request for the same key trims timestamps older than one minute. Access-log retention and deletion depend on the deployment. |

## Privacy and public pushes

- Pushes go through the project's privacy guard. Check the changed files, commit metadata, branch name and proposed PR text before publishing.
- Keep credentials, real personal or customer data, account and session details, private infrastructure identifiers, private instructions, live captions and publication records out of this public repository and its issues, PRs and logs. Keep deploy credentials in the target environment's secret store. Report exposures through [private vulnerability reporting](SECURITY.md). Revoke and rotate an exposed credential immediately.

# BalootX

## Project facts

- Repo: [BalootX/balootx-platform](https://github.com/BalootX/balootx-platform)
- Main branch: `main`
- Board (private): [BalootX](https://github.com/orgs/BalootX/projects/1)
- Release plan: [milestones page](https://github.com/BalootX/balootx-platform/milestones)
- Leads: BalootX Master, BalootX Delivery, BalootX Design
- UI project: yes
- Other repos: none
- Environments used: [CI](DELIVERY.md#ci)

## Links

- [Delivery](DELIVERY.md) and [design](DESIGN.md)
- Engineering standard v1.2 and the project-delivery-orchestration skill

## Product and current goal

BalootX combines a Saudi Baloot engine, the Foundry research dashboard, and rendered clips that show card-play decisions. The goal is to turn legally replayable synthetic scenarios into verifiable observer-safe explanations and media.

## Architecture and sources of truth

- The Python engine owns rules, transitions and replay. The [authority map](docs/architecture/authority-map.json) identifies the owner for each contract.
- The [OR-Tools Foundry guide](docs/OR_TOOLS_RESEARCH_FOUNDRY.md) covers rule and proof authority. The [Scenario V4 contract](docs/RESEARCH_SCENARIO_V4.md) covers research requests, result evidence and engine-interest criteria.
- The FastAPI service serves the [Foundry dashboard](src/baloot_platform/service/foundry_dashboard) at `/foundry`. `/studio` redirects there. The [OpenAPI contract](api) drives generated clients.
- The [video package](src/baloot_platform/video) projects the observer's replay into a render bundle for the Remotion compositions. See [the render design](DESIGN.md#rendered-clips) for the visual authorities.

## Commands

Run the Python commands at the repository root. Run npm commands in the named package.

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
| Run locally | `uvicorn baloot_platform.service.api:app --host 127.0.0.1 --port 8080`. Open `/foundry` after building the dashboard. |

## Coding rules

- Preserve the public rules and replay contracts at their owning source. Update generated views and packaged copies from that source. Use `python scripts/sync_package_resources.py` after changing mirrored resources and `python scripts/generate_openapi.py --check` to check the OpenAPI output.
- Treat a valid replay, a bounded result and an exact proof as separate evidence. Preserve a result's scope and budget when exposing it through the API, UI or render bundle.

## Risk paths

Apply the independent review in [DELIVERY.md](DELIVERY.md#merge) to changes in any of these paths.

- Every path matched by [.github/CODEOWNERS](.github/CODEOWNERS).
- Contracts and generated outputs under `api/`, `proto/`, `generated/`, `schemas/` and `src/baloot_platform/resources/schemas/`. This includes the render profile schema.
- Rendering and layout under `src/baloot_platform/video/`, `configs/video/`, `assets/` and the corresponding packaged config and asset mirrors under `src/baloot_platform/resources/`.
- Engine and proof criteria under `src/baloot_platform/{core,analysis,challenge,research,native}/` and `native/`, plus `configs/{rules,research,challenge,game,belief,match_value,challenge_contracts}/`, `configs/bundle_ortools_research_*.yaml` and their mirrors under `src/baloot_platform/resources/configs/`. This includes engine-interest scoring, viewer-seat proof and the [Scenario V4 contract](docs/RESEARCH_SCENARIO_V4.md).
- Release evidence and admission under `release/` and `src/baloot_platform/resources/release/`.
- Project definitions in `AGENTS.md`, `CLAUDE.md`, `DELIVERY.md` and `DESIGN.md`.

## Data safety

- Use synthetic fixtures for tests, previews and public examples. Enforce the observer boundary for hands, future actions and solver evidence, and follow the [adjudication boundary](docs/ADJUDICATION_RESEARCH_BOUNDARY.md) for restricted evidence.
- Preserve immutable release evidence and applied migrations. Follow the [rollback procedure](DELIVERY.md#rollback) when a shipped change must be reversed.

## Privacy and public pushes

- Pushes go through the project's privacy guard with the App identity.
- Keep credentials, real personal or customer data, account and session details, private infrastructure identifiers, private instructions, live captions and publication records out of this public repository and its issues, PRs and logs.

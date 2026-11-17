from __future__ import annotations

import json
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import uvicorn
import yaml

from baloot_platform.configuration import (
    SQLiteConfigurationStore,
    VersionedConfigurationService,
)


ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _seed(data_dir: Path) -> None:
    runtime = data_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    service = VersionedConfigurationService(
        SQLiteConfigurationStore(runtime / "configuration.sqlite3"),
        root=ROOT,
        profile=_load("configs/configuration/control-plane.yaml"),
    )
    active = service.bootstrap_packaged_bundle(actor_id="OPERATOR-E2E")
    active_components = {
        family: str(value["revisionId"])
        for family, value in active["profileRevisions"].items()
    }
    active_profiles = dict(active["profiles"])
    revisions_by_payload = {
        (family, json.dumps(payload, ensure_ascii=False, sort_keys=True)): active_components[family]
        for family, payload in active_profiles.items()
    }
    for bundle_path in (
        "configs/bundle_ortools_research_native_v3.yaml",
        "configs/bundle_ortools_research_native_v3_optional_project.yaml",
        "configs/bundle_intelligent_research.yaml",
        "configs/bundle_ortools_research_ratified_projects.yaml",
    ):
        alternate = _load(bundle_path)
        components = dict(active_components)
        for family, relative in alternate["components"].items():
            payload = _load(f"configs/{relative}")
            key = (family, json.dumps(payload, ensure_ascii=False, sort_keys=True))
            revision_id = revisions_by_payload.get(key)
            if revision_id is None:
                profile = service.create_profile(
                    family=family,
                    payload=payload,
                    created_by="OPERATOR-E2E",
                    publish=True,
                )
                revision_id = str(profile["revisionId"])
                revisions_by_payload[key] = revision_id
            components[family] = revision_id
        service.create_bundle(
            bundle_id=str(alternate["profile"]["id"]),
            version=str(alternate["profile"]["version"]),
            mode="full",
            component_revisions=components,
            created_by="OPERATOR-E2E",
            publish=True,
            allow_historical_components=True,
        )
        if alternate['profile']['id'] == 'baloot_ortools_research_native_v3_ratified_projects':
            family_components = dict(components)
    family_fixture = os.environ.get("BALOOT_OPERATOR_E2E_FAMILY_FIXTURE")
    if family_fixture in {"1", "query", "project-query"}:
        from baloot_platform.research.family_catalog import ResearchFamilyCatalogV1
        from baloot_platform.research.family_proposal_stream import build_family_target_v1
        payload = _load("configs/challenge/intelligent_research.yaml")
        payload["profile"]["status"] = "draft"
        project_query = family_fixture == 'project-query'
        catalog = ResearchFamilyCatalogV1.load(ROOT / ('configs/research/public_project_rank_inference.yaml'
            if project_query else 'configs/research/families.yaml')).to_dict()
        catalog.pop("sha256")
        family = catalog['families'][0] if project_query else next(
            row for row in catalog["families"] if row["id"] == "stopper_guard")
        family["enabled"] = True
        catalog["families"] = [family]
        target = build_family_target_v1(dict(schemaVersion=1, catalog=catalog,
            familyId=family['id'], tagQuery=({'technique': ['project-inference']} if project_query else
                {"mode": ["sun"], "trickBand": ["trick-7"]}),
            bindings=dict(suit="clubs", seat="north", rank="ace", projectType="siri",
                          contract="sun", playIndex=24), dealer="east", buyer="north", trump=None))
        policy = json.loads((ROOT / "tests/fixtures/actor_interest_calibration.json").read_text(encoding="utf-8"))["cases"]["7189"]["policy"]
        policy['baselinePolicyId'] = 'observer_safe_public_two_trick_control_v6'
        payload["scenarioResearchV4"]["qualityExecution"].update(
            candidateSource="runtime_proposal_stream_v1", proposalSeedCount=3,
            auditConstructedTrace=True, earlyPlayPolicyId="observer_safe_trick_control_v1",
            exactPlayFromTrick=7, exactTailProof=True, videoInterestingness=policy, familyTarget=target)
        if project_query:
            payload['scenarioResearchV4']['searchBudgets']['profiles']['actor_information_set_exact_v1'][
                'maximumWallSeconds'] = 60.0
        # A private, unqualified draft supplies a complete starting form; the
        # browser edits/saves it through real UI controls, never request injection.
        service.create_profile(family="challenge", payload=payload,
                               created_by="OPERATOR-E2E", publish=False)
        if family_fixture in {"query", "project-query"}:
            # Only this explicitly isolated test store publishes the opt-in fixture.
            payload["profile"]["status"] = "governed_candidate"
            catalog = target["catalog"]
            catalog.pop("sha256")
            catalog["families"][0]["contracts"] = ["sun"]
            catalog["families"][0]["capabilityTags"]["mode"] = ["sun"]
            execution = payload["scenarioResearchV4"]["qualityExecution"]
            execution.pop("familyTarget")
            execution["familyCatalog"] = ResearchFamilyCatalogV1.build(catalog).to_dict()
            profile = service.create_profile(family="challenge", payload=payload,
                created_by="OPERATOR-E2E", publish=True)
            service.create_bundle(bundle_id=('baloot_operator_project_query' if project_query else
                'baloot_operator_family_query'), version="1.0.0",
                mode="full", component_revisions={**family_components, "challenge": profile["revisionId"]},
                created_by="OPERATOR-E2E", publish=True, allow_historical_components=True)
    service.store.close()


def main() -> None:
    persistent = os.environ.get("BALOOT_OPERATOR_E2E_DATA_DIR")
    context = (
        nullcontext(persistent)
        if persistent
        else tempfile.TemporaryDirectory(prefix="balootx-foundry-operator-e2e-")
    )
    with context as temp:
        data_dir = Path(temp)
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BALOOT_DATA_DIR"] = str(data_dir)
        os.environ["BALOOT_SHUTDOWN_CONTAINER_ON_APP_EXIT"] = "true"
        os.environ.pop("BALOOT_DATABASE_URL", None)
        if not (data_dir / "runtime" / "configuration.sqlite3").exists():
            _seed(data_dir)
        from baloot_platform.service import api
        app = api.app
        preview_directory = os.environ.get("BALOOT_OPERATOR_E2E_FOUNDRY_DIR")
        if preview_directory:
            from starlette.staticfiles import StaticFiles
            preview = Path(preview_directory).resolve(strict=True)
            if not (preview / "index.html").is_file():
                raise ValueError("operator preview directory must contain index.html")
            mounts = [route for route in app.routes if getattr(route, "name", None) == "foundry-assets"]
            if len(mounts) != 1:
                raise ValueError("operator preview requires one Foundry asset mount")
            api._FOUNDRY_STATIC = preview
            mounts[0].app = StaticFiles(directory=preview)

        uvicorn.run(
            app,
            host="127.0.0.1",
            port=int(os.environ.get("BALOOT_OPERATOR_E2E_PORT", "8174")),
            log_level="warning",
        )


if __name__ == "__main__":
    main()

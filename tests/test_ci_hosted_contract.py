"""Hosted workflow policy and its fail-closed producer contract."""
import ast
from importlib.machinery import BYTECODE_SUFFIXES, EXTENSION_SUFFIXES
import json
import re
import subprocess

import pytest
import yaml

from scripts import ci_scope


@pytest.fixture
def codeowners_rules(root):
    text = (root / ".github/CODEOWNERS").read_text(encoding="utf-8")
    return [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_codeowners_covers_hosted_ci_and_release_controls(root, codeowners_rules):
    assert all(owners == ["@BalootX/maintainers"] for _, *owners in codeowners_rules)
    paths = {pattern for pattern, *_ in codeowners_rules}
    assert "/" not in paths and "*" not in paths
    workflow_scripts = {
        "/" + script
        for workflow in (root / ".github/workflows").glob("*.yml")
        for script in re.findall(r"\bscripts/[\w.]+\.py\b", workflow.read_text(encoding="utf-8"))
    }
    required = {
        "/.github/", "/ci/", "/scripts/__init__.py", "/scripts/*/", "/scripts/ci_known_failures.py",
        "/tests/conftest.py", "/tests/test_ci_hosted_contract.py",
        "/scripts/remotion_image_parity.py", "/tests/test_remotion_image_parity.py", "/tests/test_remotion_image_parity/",
        "/*.py", "/src/*.py", "/*/__init__.py", "/src/*/__init__.py",
        "/*/__init__.pyc", "/src/*/__init__.pyc", "/*/__init__*.so", "/src/*/__init__*.so",
        "/*/__init__*.pyd", "/src/*/__init__*.pyd",
        "/*.pyw", "/src/*.pyw", "/*/__init__.pyw", "/src/*/__init__.pyw",
        "/__pycache__/", "/src/__pycache__/", "/*/__pycache__/", "/src/*/__pycache__/", "/src/scripts/",
        "/*.pth", "/src/*.pth", "/pytest.ini", "/.pytest.ini", "/pytest.toml", "/.pytest.toml",
        "/setup.cfg", "/tox.ini", "/tests/setup.cfg", "/tests/tox.ini", "/tests/pyproject.toml",
        "/tests/pytest.ini", "/tests/.pytest.ini", "/tests/pytest.toml", "/tests/.pytest.toml",
        "/tests/deterministic_work.py", "/tests/conftest/", "/tests/deterministic_work/",
        "/tests/test_ci_hosted_contract/", "/tests/test_repository_structure/", "/tests/test_media_render_artifact/",
        "/tests/__pycache__/", "**/.npmrc",
        "/src/baloot_platform/service/foundry_dashboard/scripts/check-generated-api.mjs",
        "/src/baloot_platform/video/remotion/scripts/", "/scripts/render_remotion_plugin_parity.py",
        "/scripts/render_research_notice.py", "/scripts/render_research_scenario_v4_goldens.py",
        "/scripts/verify_professional_flagship.py", "/scripts/verify_professional_decision_clip.py",
        "/scripts/verify_professional_video_prototypes.py",
        "/scripts/build_reproducible_wheel.py", "/scripts/package_release*.py", "/scripts/verify_release*.py",
        "/scripts/check_release_readiness.py", "/scripts/migrate_postgres.py", "/migrations/", "/Dockerfile",
        "/pyproject.toml", "/sdk/go/go.mod", "/sdk/typescript/package.json",
        "/src/baloot_platform/service/foundry_dashboard/package.json",
        "/src/baloot_platform/service/foundry_dashboard/package-lock.json",
        "/src/baloot_platform/video/remotion/package.json", "/src/baloot_platform/video/remotion/package-lock.json",
    }
    suffixes = {".so", ".pyd", ".pyc"}  # Linux and Windows, including ABI-tagged variants.
    suffixes.update("." + suffix.rsplit(".", 1)[1] for suffix in (*EXTENSION_SUFFIXES, *BYTECODE_SUFFIXES))
    required.update(f"/{scope}*{suffix}" for scope in ("", "scripts/", "src/", "tests/") for suffix in suffixes)
    assert required | workflow_scripts <= paths


def test_codeowners_matches_gate_import_package_and_non_source_shadows(tmp_path, codeowners_rules):
    # These paths are evaluated as Git patterns without creating or importing modules.
    paths = [
        "scripts/ci_known_failures/__init__.py",
        "scripts/__pycache__/ci_known_failures.cpython-312.pyc",
        "scripts/run_heavy_pytest.abi3.so",
        "scripts/run_isolated_pytest.cp313-win_amd64.pyd",
        "scripts/ci_known_failures.pyc",
        "src/scripts/ci_known_failures/__init__.py",
        "tests/deterministic_work/__init__.py",
        "tests/.pytest.toml",
        ".npmrc",
        "src/baloot_platform/video/remotion/.npmrc",
    ]
    subprocess.run(["git", "init", "-q", "--template=", str(tmp_path)], check=True, timeout=10)
    patterns = [pattern for pattern, *owners in codeowners_rules if owners == ["@BalootX/maintainers"]]
    (tmp_path / ".gitignore").write_text("\n".join(patterns) + "\n", encoding="utf-8")
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=tmp_path, input="\0".join([*paths, "scripts/benchmark_agent_symmetry.py", ""]).encode(),
        capture_output=True, check=True, timeout=10,
    )
    assert result.stdout.decode().split("\0") == [*paths, ""]


def test_ci_shard_verifiers_use_python_safe_path(root):
    source = ast.parse((root / "scripts/render_release.py").read_bytes())
    shard = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "render_ci_shard")
    commands = [
        node.elts for node in ast.walk(shard)
        if isinstance(node, ast.List) and node.elts and ast.unparse(node.elts[0]) == "sys.executable"
    ]
    assert len(commands) == 3
    assert all(len(command) >= 3 and ast.unparse(command[1]) == "'-P'" for command in commands)


def test_hosted_ci_contract(root):
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())
    triggers = workflow[True]  # PyYAML's YAML 1.1 spelling of on.
    assert triggers["push"] == {"branches": ["main"]}
    assert triggers["merge_group"] == {"types": ["checks_requested"], "branches": ["main"]}
    assert "pull_request_target" not in triggers
    assert not {"paths", "paths-ignore"} & triggers["pull_request"].keys()
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    main_push = "github.event_name == 'push' && github.ref == 'refs/heads/main' && "
    for name, job in jobs.items():
        assert job["runs-on"] == "ubuntu-24.04"
        assert job["timeout-minutes"] > 0
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[0-9a-f]{40}", step["uses"].rsplit("@", 1)[-1])
                if step["uses"].startswith("actions/checkout@"):
                    assert step["with"]["persist-credentials"] is False
                    assert not {"ref", "repository"} & step["with"].keys(), name
                if step["uses"].startswith(("actions/cache/save@", "actions/cache@")):
                    condition = step.get("if") or ""
                    assert condition.startswith(main_push) and "||" not in condition, name
            assert not step.get("continue-on-error")
    assert jobs["ci-gate"]["name"] == "${{ github.event_name == 'workflow_dispatch' && 'CI diagnostic result' || 'ci-gate' }}"
    assert jobs["plan"]["if"] == (
        "${{ !inputs.viewer-seat-proof && (github.event_name != 'pull_request' || !github.event.pull_request.draft) }}"
    )
    assert jobs["plan"]["steps"][0]["with"]["fetch-depth"] == 2
    assert jobs["plan"]["steps"][-1]["env"] == {
        "PR_BASE": "${{ github.event.pull_request.base.sha }}",
        "PR_HEAD": "${{ github.event.pull_request.head.sha }}",
    }
    assert jobs["dependency-review"]["if"] == "github.event_name == 'pull_request'"
    smoke = jobs["python-version-smoke"]
    assert smoke["needs"] == "plan"
    assert smoke["if"] == "github.event_name == 'pull_request'"
    assert smoke["steps"][1]["with"]["python-version"] == "3.12"
    assert "python -m pytest" in smoke["steps"][-1]["run"]
    assert set(jobs["python-affected"]["needs"]) == {"plan", *ci_scope.TEST_SETUP}
    for job in ci_scope.FULL:
        predicate = "needs.plan.outputs.full == 'true'"
        if job in ci_scope.TEST_SETUP:
            predicate += " || needs.plan.outputs.tests != '[]'"
        assert jobs[job]["if"] == predicate
    gate_step = jobs["ci-gate"]["steps"][-1]
    assert gate_step["env"]["IS_DRAFT"] == "${{ github.event_name == 'pull_request' && github.event.pull_request.draft }}"
    draft_guard, gate_command = gate_step["run"].split("python3 -P scripts/ci_scope.py gate")
    assert "exit 1" in draft_guard and not gate_command.strip()
    expected_producers = ci_scope.FULL | ci_scope.SCOPED.keys() | {
        "plan", "preflight", "dependency-review", "python-version-smoke", "python-affected",
    }
    assert set(jobs["ci-gate"]["needs"]) == expected_producers == jobs.keys() - {
        "ci-gate", "legacy-ci-result", "viewer-seat-proof",
    }
    assert jobs["ci-gate"]["if"] == "${{ always() && !inputs.viewer-seat-proof }}"
    legacy = jobs["legacy-ci-result"]
    assert legacy["name"] == "${{ github.event_name == 'workflow_dispatch' && 'CI diagnostic compatibility' || 'CI result' }}"
    assert legacy["needs"] == "ci-gate"
    assert legacy["if"] == "${{ always() && github.event_name != 'workflow_dispatch' }}"
    assert legacy["steps"] == [{
        "env": {"CI_GATE_RESULT": "${{ needs.ci-gate.result }}"},
        "run": 'test "$CI_GATE_RESULT" = success',
    }]
    for name in (
        "plan", "ci-gate", "python-coverage", "python-fast-run", "python-backend-run", "python-heavy",
        "media-inputs", "media-render", "media-release-smoke", "viewer-seat-proof", "remotion-image-parity",
    ):
        commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
        flags = re.findall(r"\bpython3? ([^;\n]*?)scripts/[\w.]+\.py\b", commands)
        assert flags and all("-P" in options.split() for options in flags), name


def test_workflow_only_cancels_superseded_runs_of_the_same_pr(root):
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())
    # A PR group has its PR number and a stable suffix. Push, dispatch and merge
    # group events include their unique run ID, protecting pending runs as well.
    assert workflow["concurrency"] == {
        "group": (
            "ci-${{ github.event_name }}-${{ github.event.pull_request.number || github.ref }}-"
            "${{ github.event_name != 'pull_request' && github.run_id || 'pr' }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    assert all("concurrency" not in job for job in workflow["jobs"].values())


def test_remotion_image_parity_builds_and_checks_the_video_worker(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    parity = jobs["remotion-image-parity"]
    assert parity["needs"] == "plan"
    assert parity["if"] == "needs.plan.outputs.video_worker == 'true'"
    assert jobs["plan"]["outputs"]["video_worker"] == "${{ steps.plan.outputs.video_worker }}"
    assert "remotion-image-parity" in jobs["media-render"]["needs"]
    checkout, restore, build, check, save = parity["steps"]
    assert checkout["with"]["persist-credentials"] is False
    assert restore["with"] == save["with"]
    assert all(path in restore["with"]["key"] for path in ("Dockerfile", ".dockerignore", "pyproject.toml", "remotion/package*.json"))
    assert "--target video-worker --load" in build["run"] and "--push" not in build["run"]
    assert build["env"]["MAIN_PUSH"] == "${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}"
    assert "elif [[ \"$MAIN_PUSH\" == 'true' ]]; then\n  cache_args+=(--cache-to " in build["run"]
    assert "docker run --rm --network none" in check["run"]
    assert "sha256sum --check --strict /opt/baloot-remotion/.lock-sha256" in check["run"]
    assert "python -P scripts/remotion_image_parity.py /opt/baloot-remotion/node_modules package-lock.json" in check["run"]
    assert "target=/check/package-lock.json,readonly" in check["run"]
    assert "target=/check/scripts/remotion_image_parity.py,readonly" in check["run"]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert re.fullmatch(r"FROM python:3\.13-slim@sha256:[0-9a-f]{64} AS python-base", dockerfile.splitlines()[0])
    assert jobs["python-wheelhouse"]["strategy"]["matrix"]["python"] == jobs["python-fast-run"]["strategy"]["matrix"]["python"]
    assert '"3.13"' in jobs["python-wheelhouse"]["strategy"]["matrix"]["python"]
    assert "sha256sum package-lock.json > .lock-sha256" in dockerfile
    assert "COPY --from=remotion-runtime /remotion/.lock-sha256 /opt/baloot-remotion/.lock-sha256" in dockerfile


def test_viewer_seat_proof_dispatch_uses_pinned_fixtures_and_skips_ci(root):
    import hashlib

    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())
    option = workflow[True]["workflow_dispatch"]["inputs"]["viewer-seat-proof"]
    assert option["type"] == "boolean" and option["default"] is False
    jobs = workflow["jobs"]
    assert jobs["preflight"]["if"] == "${{ !inputs.viewer-seat-proof }}"
    proof = jobs["viewer-seat-proof"]
    assert proof["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.viewer-seat-proof }}"
    assert "needs" not in proof
    assert {name for name, job in jobs.items() if "needs" not in job} == {
        "plan", "preflight", "dependency-review", "viewer-seat-proof",
    }
    cohort = proof["strategy"]["matrix"]["include"]
    assert {row["clip"] for row in cohort} == {"16-05", "19-03", "18-03"}
    for row in cohort:
        path = root / f"examples/research-scenario-v4-single-winner-202611{row['clip']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert "steps.proof.outcome != 'skipped'" in proof["steps"][-1]["if"]


def test_full_ci_installs_pinned_ffmpeg_once_per_consumer(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    consumers = {
        "media-inputs", "media-render", "media-release-smoke",
        "python-fast-run", "python-backend-run", "python-heavy", "python-affected",
    }
    media_steps = jobs["media-render"]["steps"]
    download = next(step for step in media_steps if step.get("name") == "Download verified FFmpeg packages")
    install = next(step for step in media_steps if step.get("name") == "Install pinned FFmpeg")
    for name, job in jobs.items():
        steps = job["steps"]
        expected = int(name in consumers)
        assert sum(step is download for step in steps) == expected, name
        assert sum(step is install for step in steps) == expected, name
        if expected:
            assert steps.index(download) < steps.index(install), name
            assert all("apt-get update" not in step.get("run", "") for step in steps), name
    assert "sha256sum --check --strict" in install["run"]
    assert "dpkg --skip-same-version --install" in install["run"]
    assert "ffmpeg-packages" in jobs["media-inputs"]["needs"]
    assert jobs["media-render"]["timeout-minutes"] == 25


@pytest.mark.parametrize("event", ["pull_request", "push", "workflow_dispatch", "merge_group"])
def test_gate_rejects_unproven_full_producers(event):
    plan = {key: json.dumps(value) for key, value in ci_scope.current_plan().items()}
    results = {job: {"result": "success"} for job in ci_scope.FULL | ci_scope.SCOPED.keys()
               | {"plan", "preflight", "dependency-review", "python-affected", "python-version-smoke"}}
    results["plan"]["outputs"] = plan
    results["dependency-review"]["result"] = ("success" if event == "pull_request" else "skipped")
    results["python-affected"]["result"] = "skipped"
    results["python-version-smoke"]["result"] = results["dependency-review"]["result"]
    ci_scope.validate_results(event, results)
    with pytest.raises(ValueError, match="its outputs are unavailable"):
        ci_scope.validate_results(event, {**results, "plan": {"result": "success"}})
    with pytest.raises(ValueError, match="CI producers differ from the plan"):
        ci_scope.validate_results(event, {**results, "extra-job": {"result": "success"}})
    for job in results:
        old = results[job]["result"]
        for bad in {"failure", "cancelled", "skipped", "success", ""} - {old}:
            results[job]["result"] = bad
            with pytest.raises(ValueError):
                ci_scope.validate_results(event, results)
        results[job]["result"] = old
    blocked_backend = {**results, "foundry": {"result": "failure"},
                       "python-backend-run": {"result": "skipped"}, "python-coverage": {"result": "skipped"}}
    with pytest.raises(ValueError, match="CI producers differ from the plan"):
        ci_scope.validate_results(event, blocked_backend)
    for key in ("full", *ci_scope.SCOPED.values()):
        plan[key] = "false"
        with pytest.raises(ValueError):
            ci_scope.validate_results(event, results)
        plan[key] = "true"
    results.pop("python-heavy")
    with pytest.raises(ValueError):
        ci_scope.validate_results(event, results)


@pytest.mark.parametrize(("paths", "producers"), [
    ({f"docs/{__name__}.md"}, set()),
    ({ci_scope.FOUNDRY + "main.tsx"}, {"foundry", "remotion-image-parity", "python-affected", *ci_scope.TEST_SETUP}),
    ({"sdk/go/client.go"}, {"sdks", "python-affected", *ci_scope.TEST_SETUP}),
    ({f"docs/{__name__}.md", ci_scope.FOUNDRY + "main.tsx", "sdk/typescript/src/client.ts"},
     {"foundry", "sdks", "remotion-image-parity", "python-affected", *ci_scope.TEST_SETUP}),
])
def test_gate_requires_exact_scoped_producers(root, paths, producers):
    selection = ci_scope.scoped_plan(paths)
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    required = producers | {"plan", "preflight", "dependency-review", "python-version-smoke"}
    results = {job: {"result": "success" if job in required else "skipped"} for job in jobs["ci-gate"]["needs"]}
    results["plan"]["outputs"] = {key: json.dumps(value) for key, value in selection.items()}
    ci_scope.validate_results("pull_request", results)
    assert sum(len(jobs[job].get("strategy", {}).get("matrix", {}).get("python", [None]))
               for job in required | {"ci-gate"}) <= 20
    for job, details in results.items():
        status = details["result"]
        for bad in {"cancelled", "failure", "skipped", "success"} - {status}:
            details["result"] = bad
            with pytest.raises(ValueError):
                ci_scope.validate_results("pull_request", results)
        details["result"] = status
    for event in ("push", "merge_group"):
        with pytest.raises(ValueError, match="full CI plan is required"):
            ci_scope.validate_results(event, results)


@pytest.mark.parametrize("path", [
    ".github/workflows/ci.yml", "ci/TIMINGS.md", "scripts/ci_scope.py", "pyproject.toml",
    "sdk/go/go.mod", "sdk/typescript/package.json", ci_scope.FOUNDRY + "../package-lock.json",
    "tests/conftest.py", "tests/test_ci_hosted_contract.py", "tests/test_core.py",
    "tests/test_research_ortools_unified_executor.py", "tests/test_unmeasured.py",
    "docs/architecture/authority-map.json", "src/baloot_platform/core/rules.py",
    "Dockerfile", ".dockerignore", "scripts/remotion_image_parity.py", "tests/test_remotion_image_parity.py",
    "src/baloot_platform/video/remotion/package.json", "src/baloot_platform/video/remotion/package-lock.json",
    "LICENSE", "THIRD-PARTY-NOTICES", "native/CMakeLists.txt", "configs/runtime/docker_v1.yaml",
    "schemas/version.schema.json", "generated/openapi.json", "proto/baloot.proto", "migrations/001_initial.sql",
    "assets/fonts/manifest.json", "release/saudi-baloot-official-source-verification-20260831-v2.json",
])
def test_unknown_shared_and_ci_paths_force_the_full_profile(monkeypatch, path):
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {"README.md", path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


def test_plan_falls_back_for_documentation_anchor(root, monkeypatch):
    profile = yaml.safe_load((root / "configs/adjudication/research_events_v1.yaml").read_bytes())
    path = profile["events"]["out_of_turn"]["documentation"].split("#")[0]
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


def test_plan_falls_back_for_cross_test_path(root, monkeypatch):
    acceptance = (root / "tests/test_research_ortools_authority_acceptance.py").read_bytes()
    match = re.search(rb'"(tests/[^"]+\.py)::test_preference_miss_does_not_reject', acceptance)
    assert match
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {match[1].decode()})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


@pytest.mark.parametrize(("path", "consumer", "reference"), [
    ("docs/example.md", "tests/nested/guide.json", '["docs/example.md#anchor"]'),
    ("docs/example.md", "tests/test_reader.py", 'root / "docs" / "example.md"'),
    ("tests/test_mcp_builtin.py", "src/owner.py", '"tests/test_mcp_builtin.py::test_mcp_stdio_process"'),
])
def test_repository_reference_forces_full(path, consumer, reference, tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, timeout=10)
    target = tmp_path / consumer
    target.parent.mkdir(parents=True)
    target.write_text(reference, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", consumer], check=True, timeout=10)
    monkeypatch.setattr(ci_scope, "ROOT", tmp_path)
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


def test_reference_scan_error_forces_full(monkeypatch):
    def fail(*args):
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(ci_scope, "git", fail)
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {"docs/unavailable.md"})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


def test_plan_runs_selected_tests_on_all_python_versions(root):
    selection = ci_scope.scoped_plan({ci_scope.FOUNDRY + "main.tsx"})
    assert selection and not selection["full"] and selection["tests"]
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    affected = jobs["python-affected"]
    strategy = affected.get("strategy", {})
    assert strategy.get("matrix", {}).get("python") == list(ci_scope.VERSIONS)
    assert strategy["fail-fast"] is False
    assert affected["if"] == "needs.plan.outputs.tests != '[]'"
    assert affected["steps"][1]["with"]["python-version"] == "${{ matrix.python }}"
    assert any(step.get("env", {}).get("AFFECTED_TESTS") == "${{ needs.plan.outputs.tests }}"
               for step in affected["steps"])
    assert "python-affected" in jobs["ci-gate"]["needs"]


def test_only_exact_pr_merge_diff_can_select_scoped_work(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], timeout=10)

    git("init", "-q", "-b", "base")
    git("config", "user.name", "CI scope fixture")
    git("config", "user.email", "ci-fixture@users.noreply.github.com")
    git("commit", "--allow-empty", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    git("switch", "-qc", "pr")
    (tmp_path / "README.md").write_text("Documentation\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-qm", "docs")
    head = git("rev-parse", "HEAD").decode().strip()
    git("switch", "-q", "base")
    git("merge", "-q", "--no-ff", "pr", "-m", "PR merge")
    monkeypatch.setattr(ci_scope, "git", git)

    assert ci_scope.changed_paths(base, head) == {"README.md"}
    assert ci_scope.current_plan("pull_request", base, head)["full"] is False
    full = ci_scope.current_plan()
    assert ci_scope.current_plan("pull_request", head, base) == full
    assert ci_scope.current_plan("push", base, head) == full
    assert ci_scope.current_plan("workflow_dispatch", base, head) == full
    assert ci_scope.current_plan("merge_group", base, head) == full
    monkeypatch.setattr(ci_scope, "git", lambda *_: git("diff", "absent-ref"))
    assert ci_scope.current_plan("pull_request", base, head) == full


def test_merge_group_plan_binds_checkout_and_event_context(tmp_path, monkeypatch):
    ref = "refs/heads/gh-readonly-queue/main/pr-104-example"
    base_sha, head_sha = "a" * 40, "b" * 40
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"merge_group": {
        "head_ref": ref, "head_sha": head_sha, "base_ref": "refs/heads/main", "base_sha": base_sha,
    }}), encoding="utf-8")
    outputs = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "merge_group")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REF", ref)
    monkeypatch.setenv("GITHUB_SHA", head_sha)
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setattr(ci_scope.sys, "argv", ["ci_scope.py", "plan"])

    def checkout(*args):
        assert args == ("rev-parse", "HEAD")
        return head_sha.encode("ascii") + b"\n"

    monkeypatch.setattr(ci_scope, "git", checkout)
    ci_scope.main()
    pairs = dict(line.split("=", 1) for line in outputs.read_text(encoding="utf-8").splitlines())
    assert {key: json.loads(value) for key, value in pairs.items()} == ci_scope.current_plan()
    outputs.unlink()
    for key, wrong in (("GITHUB_REF", "refs/heads/main"), ("GITHUB_SHA", base_sha)):
        with monkeypatch.context() as patch, pytest.raises(ValueError, match="does not match the merge group head"):
            patch.setenv(key, wrong)
            ci_scope.main()
    monkeypatch.setattr(ci_scope, "git", lambda *_: base_sha.encode("ascii"))
    with pytest.raises(ValueError, match="does not match the merge group head"):
        ci_scope.main()
    assert not outputs.exists()


@pytest.mark.parametrize("delta", [
    b"", b":100644 000000 1111111 0000000 D\0README.md\0",
    b":100644 120000 1111111 2222222 T\0README.md\0",
    b":100644 100644 1111111 2222222 M\0docs/with a space.md\0",
    b":100644 100644 1111111 2222222 M\0README.md",
])
def test_ambiguous_or_empty_git_diff_runs_full(monkeypatch, delta):
    base, head = "a" * 40, "b" * 40
    parents = f"{'c' * 40} {base} {head}\n".encode()
    monkeypatch.setattr(ci_scope, "git", lambda command, *_: parents if command == "rev-list" else delta)
    assert ci_scope.current_plan("pull_request", base, head) == ci_scope.current_plan()


def test_one_run_never_holds_more_than_twenty_hosted_jobs(root):
    # Shared Actions capacity: the largest set of mutually independent jobs, weighted by matrix width.
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    needs = {name: set([job["needs"]] if isinstance(job.get("needs"), str) else job.get("needs", ())) for name, job in jobs.items()}
    def ancestors(name):
        found = set(needs[name])
        for parent in needs[name]:
            found |= ancestors(parent)
        return found
    before = {name: ancestors(name) for name in jobs}
    def width(job, is_pr):
        strategy = job.get("strategy") or {}
        size = 1
        matrix = strategy.get("matrix") or {}
        for key, values in matrix.items():
            if isinstance(values, list):
                size *= len(values)
            elif key == "python":
                size *= 1 if is_pr else len(ci_scope.VERSIONS)
        return min(size, strategy.get("max-parallel", size))
    def peak(index, chosen, names, is_pr):
        if index == len(names):
            return 0
        name = names[index]
        best = peak(index + 1, chosen, names, is_pr)
        if all(name not in before[other] and other not in before[name] for other in chosen):
            best = max(best, width(jobs[name], is_pr) + peak(index + 1, chosen | {name}, names, is_pr))
        return best
    full = jobs.keys() - {"python-affected", "viewer-seat-proof"}
    assert peak(0, frozenset(), sorted(full), True) <= 20
    assert peak(0, frozenset(), sorted(full - {"dependency-review", "python-version-smoke"}), False) <= 20
    assert peak(0, frozenset(), ["viewer-seat-proof"], False) <= 20


def test_dependabot_groups_weekly_version_minor_and_patch_updates(root):
    updates = yaml.safe_load((root / ".github/dependabot.yml").read_bytes())["updates"]
    assert {row["package-ecosystem"] for row in updates} == {"github-actions", "pip", "npm", "gomod", "docker"}
    for update in updates:
        assert update["schedule"]["interval"] == "weekly"
        group, = update["groups"].values()
        assert group == {"patterns": ["*"], "update-types": ["minor", "patch"]}

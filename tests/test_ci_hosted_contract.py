"""Hosted workflow policy and its fail-closed producer contract."""
import ast
from importlib.machinery import BYTECODE_SUFFIXES, EXTENSION_SUFFIXES
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from ci import critical_review
from scripts import ci_scope
from scripts import run_required_pytest
import test_v20_asan_verifier_portability as asan_portability


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
        "/scripts/run_required_pytest.py", "/scripts/run_local_capped.ps1",
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
        "/scripts/render_kickoff.py", "/scripts/render-kickoff.json", "/scripts/render-kickoff.schema.json",
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
        if name == "viewer-seat-proof":
            continue  # Checked with the nightly caller and shared workflow below.
        assert job["runs-on"] == ("windows-latest" if name == "python-windows" else "ubuntu-24.04"), name
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
        "ci-gate", "viewer-seat-proof",
    }
    assert jobs["ci-gate"]["if"] == "${{ always() && !inputs.viewer-seat-proof }}"
    assert jobs["ci-gate"]["permissions"] == {"contents": "read", "pull-requests": "read"}
    review_step = next(step for step in jobs["ci-gate"]["steps"] if "GH_TOKEN" in step.get("env", {}))
    assert review_step["if"] == (
        "(github.event_name == 'pull_request' && !github.event.pull_request.draft) || github.event_name == 'merge_group'"
    )
    assert review_step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert 'git show "$REVIEW_BASE:ci/critical_review.py"' in review_step["run"]
    assert 'python3 -I "$RUNNER_TEMP/critical-review.py"' in review_step["run"]
    assert "python3 -I ci/critical_review.py" not in review_step["run"]
    assert not re.search(r"^\s*exit\s+0\s*$", review_step["run"], re.MULTILINE)
    for name in (
        "plan", "ci-gate", "python-coverage", "python-fast-run", "python-backend-run", "python-heavy",
        "media-inputs", "media-render", "media-release-smoke", "remotion-image-parity",
    ):
        commands = "\n".join(step.get("run", "") for step in jobs[name]["steps"])
        flags = re.findall(r"\bpython3? ([^;\n]*?)scripts/[\w.]+\.py\b", commands)
        assert flags and all("-P" in options.split() for options in flags), name


@pytest.fixture
def review_history(tmp_path, monkeypatch, root):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, timeout=10).stdout

    git("init", "-q", "-b", "main")
    git("config", "user.name", "CI review fixture")
    git("config", "user.email", "ci-fixture@users.noreply.github.com")
    git("config", "commit.gpgsign", "false")
    git("remote", "add", "origin", ".")
    policy = tmp_path / critical_review.POLICY_PATH
    policy.parent.mkdir()
    policy.write_bytes((root / critical_review.POLICY_PATH).read_bytes())
    kickoff = tmp_path / "scripts/render_kickoff.py"
    kickoff.parent.mkdir()
    kickoff.write_text("base content\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(critical_review, "git", git)

    def commit(path, followup=None, remove=False, squash=False):
        git("switch", "-qc", "topic")
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic change\n", encoding="utf-8")
        git("add", ".")
        git("commit", "-qm", "change")
        reviewed = git("rev-parse", "HEAD").decode().strip()
        if followup is not None or remove:
            if remove:
                target.unlink()
            else:
                (tmp_path / followup).write_text("later change\n", encoding="utf-8")
            git("add", "-A")
            git("commit", "-qm", "followup")
        head = git("rev-parse", "HEAD").decode().strip()
        git("switch", "-q", "main")
        if squash:
            git("merge", "--squash", "topic")
            git("commit", "-qm", "queued change")
        else:
            git("merge", "--no-ff", "topic", "-m", "PR merge")
        return base, reviewed, head, git("rev-parse", "HEAD").decode().strip()

    return commit


@pytest.mark.parametrize(("path", "followup", "remove", "records", "other_verdict", "error"), [
    ("scripts/render_kickoff.py", None, False, (), False, "needs a VERDICT"),
    ("scripts/render_kickoff.py", None, False, (), True, "needs a VERDICT"),
    ("scripts/render_kickoff.py", None, False, ("VERDICT: GO {head}",), False, None),
    ("scripts/render_kickoff.py", "scripts/render_kickoff.py", False, ("VERDICT: GO {reviewed}",), False, "changed or missing"),
    ("scripts/render_kickoff.py", "README.md", False, ("VERDICT: GO {reviewed}",), False, None),
    ("scripts/render_kickoff.py", None, True, ("VERDICT: GO {reviewed}",), False, "changed or missing"),
    ("scripts/render_kickoff.py", None, True, ("VERDICT: GO {head}",), False, None),
    ("scripts/render-kickoff.json", None, False, ("VERDICT: GO {head}", "VERDICT: NO_GO {head}"), False, "NO_GO verdict"),
    ("scripts/render-kickoff.json", None, False, ("VERDICT: NO_GO {head}", "VERDICT: GO {head}"), False, None),
    ("scripts/render-kickoff.json", None, False, ("VERDICT: GO {head}", "VERDICT: GO ffffffff"), False, "malformed VERDICT"),
    ("scripts/render-kickoff.json", None, False, ("PR #104 verdict GO on {head}",), False, "needs a VERDICT"),
    ("scripts/render-kickoff.schema.json", None, False, (), False, "needs a VERDICT"),
    ("ci/critical-paths.json", None, False, (), False, "needs a VERDICT"),
    ("README.md", None, False, (), False, None),
])
def test_critical_review_on_pull_request(review_history, monkeypatch, path, followup, remove, records, other_verdict, error):
    base, reviewed, head, candidate = review_history(path, followup=followup, remove=remove)
    reviewer_ids, _ = critical_review.policy(base)
    author = {"id": next(iter(reviewer_ids))}
    repo = "test/example"
    event = {"pull_request": {"number": 104, "base": {"sha": base}, "head": {"sha": head}}}
    endpoint = f"repos/{repo}/pulls/104"
    files = [{"filename": path, "status": "removed" if remove else "modified"}]
    if followup and followup != path:
        files.append({"filename": followup, "status": "added"})

    def github(*args):
        if args == (endpoint,):
            return {**event["pull_request"], "state": "open", "changed_files": len(files)}
        if args == ("--paginate", "--slurp", f"{endpoint}/files?per_page=100"):
            return [files]
        assert path != "README.md"
        assert args == ("--paginate", "--slurp", f"repos/{repo}/issues/104/comments?per_page=100")
        pages = [
            [{"id": index, "user": author, "body": record.format(head=head, reviewed=reviewed)}]
            for index, record in enumerate(records, start=1)
        ]
        if other_verdict:
            pages.append([{"user": {"id": max(reviewer_ids) + 1}, "body": f"VERDICT: GO {head}"}])
        return pages

    monkeypatch.setattr(critical_review, "github", github)
    args = ("pull_request", event, repo, "refs/pull/104/merge", candidate)
    if error:
        with pytest.raises(ValueError, match=error):
            critical_review.check(*args)
    else:
        critical_review.check(*args)


def test_critical_review_binds_queue_entry_to_pr_head(review_history, monkeypatch):
    base, _, head, candidate = review_history("scripts/render-kickoff.json", squash=True)
    reviewer_ids, _ = critical_review.policy(base)
    ref = f"refs/heads/gh-readonly-queue/main/pr-104-{base}"
    event = {"merge_group": {"base_ref": "refs/heads/main", "base_sha": base, "head_ref": ref, "head_sha": candidate}}
    entry = {"baseCommit": {"oid": base}, "headCommit": {"oid": candidate}}
    pr = {"mergeQueueEntry": entry, "headRefOid": head}
    comment = {"user": {"id": next(iter(reviewer_ids))}, "body": f"VERDICT: GO {base}"}
    snapshot = {"number": 104, "state": "open", "head": {"sha": head}, "base": {"sha": base}, "changed_files": 1}
    files = [{"filename": "scripts/render-kickoff.json", "status": "added"}]

    def github(*args):
        if args[0] == "graphql":
            assert args[-6:] == ("-f", "owner=test", "-f", "name=example", "-F", "number=104")
            return {"data": {"repository": {"pullRequest": pr}}}
        if args == ("repos/test/example/pulls/104",):
            return snapshot
        if args == ("--paginate", "--slurp", "repos/test/example/pulls/104/files?per_page=100"):
            return [files]
        assert args == ("--paginate", "--slurp", "repos/test/example/issues/104/comments?per_page=100")
        assert files[0]["filename"] != "README.md"
        return [[comment]]

    monkeypatch.setattr(critical_review, "github", github)
    args = ("merge_group", event, "test/example", ref, candidate)
    with pytest.raises(ValueError, match="changed or missing"):
        critical_review.check(*args)
    comment["body"] = f"VERDICT: GO {head}"
    critical_review.check(*args)
    files[0]["filename"] = "ci/fixture.json"
    with pytest.raises(ValueError, match="absent from both the PR base and head"):
        critical_review.check(*args)
    files[0]["filename"] = "README.md"
    critical_review.check(*args)
    snapshot["changed_files"] = 2
    with pytest.raises(ValueError, match="snapshot is incomplete"):
        critical_review.check(*args)
    entry["headCommit"]["oid"] = "f" * 40
    with pytest.raises(ValueError, match="no longer has the event's merge-queue entry"):
        critical_review.check(*args)
    with pytest.raises(ValueError, match="Merge-group ref"):
        critical_review.check("merge_group", event, "test/example", "refs/heads/main", candidate)


def _assert_single_ci_check_producers(workflows):
    names = {
        (path.name, job_id): str(job.get("name") or job_id)
        for path in sorted(workflows.iterdir())
        if path.is_file() and path.suffix in {".yml", ".yaml"}
        for job_id, job in yaml.safe_load(path.read_bytes())["jobs"].items()
    }
    expected = {
        "ci-gate": [("ci.yml", "ci-gate")],
        "CI result": [],
    }
    for check, producers in expected.items():
        actual = [source for source, name in names.items() if check in name]
        assert actual == producers, f"{check} producers: expected {producers}, found {actual}"


def test_ci_check_names_have_single_producers(root):
    _assert_single_ci_check_producers(root / ".github/workflows")


def test_ci_check_producer_contract_rejects_duplicate_workflow_job(root, tmp_path):
    (tmp_path / "ci.yml").write_bytes((root / ".github/workflows/ci.yml").read_bytes())
    (tmp_path / "extra.yml").write_text(
        "on: push\njobs:\n  ci-gate:\n    runs-on: ubuntu-24.04\n    steps:\n      - run: exit 0\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="ci-gate producers"):
        _assert_single_ci_check_producers(tmp_path)


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
    checkout, restore, build, check, check_checkout, save = parity["steps"]
    assert checkout["with"]["persist-credentials"] is False
    assert restore["with"] == save["with"]
    assert all(path in restore["with"]["key"] for path in (
        "Dockerfile", "ci/Dockerfile.buildkit", ".dockerignore", "pyproject.toml",
        "requirements-docker.txt", "remotion/package*.json",
    ))
    assert "--target video-worker --load" in build["run"] and "--push" not in build["run"]
    assert build["env"]["MAIN_PUSH"] == "${{ github.event_name == 'push' && github.ref == 'refs/heads/main' }}"
    assert "elif [[ \"$MAIN_PUSH\" == 'true' ]]; then\n  cache_args+=(--cache-to " in build["run"]
    assert "docker run --rm --network none" in check["run"]
    assert "sha256sum --check --strict /opt/baloot-remotion/.lock-sha256" in check["run"]
    assert "python -P scripts/remotion_image_parity.py /opt/baloot-remotion/node_modules package-lock.json" in check["run"]
    assert "target=/check/package-lock.json,readonly" in check["run"]
    assert "target=/check/scripts/remotion_image_parity.py,readonly" in check["run"]
    assert "sudo chown -R 10001" in check_checkout["run"]
    assert "docker run --rm --network none --read-only --workdir /checkout" in check_checkout["run"]
    assert "target=/checkout,readonly" in check_checkout["run"]
    assert "target=/receipts,readonly" in check_checkout["run"]
    assert 'test "$(id -u)" -eq 10001' in check_checkout["run"]
    assert "notice.main(public=True, source_head=head, source_tree=tree)" in check_checkout["run"]
    assert 'run_producer("wrong", source_tree, source_tree)' in check_checkout["run"]
    assert 'notice.STYLE.parents[1] / "node_modules"' in check_checkout["run"]
    assert "safe.directory" not in check_checkout["run"]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert re.fullmatch(r"FROM python:3\.13-slim@sha256:[0-9a-f]{64} AS python-base", dockerfile.splitlines()[0])
    assert jobs["python-wheelhouse"]["strategy"]["matrix"]["python"] == jobs["python-fast-run"]["strategy"]["matrix"]["python"]
    assert '"3.13"' in jobs["python-wheelhouse"]["strategy"]["matrix"]["python"]
    assert "sha256sum package-lock.json > .lock-sha256" in dockerfile
    assert "COPY --from=remotion-runtime /remotion/.lock-sha256 /opt/baloot-remotion/.lock-sha256" in dockerfile


def test_render_image_inputs_are_pinned(root):
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    stages = set()
    for instruction in (line for line in dockerfile.splitlines() if line.startswith("FROM ")):
        source, *alias = instruction.split()[1:]
        assert source in stages or re.fullmatch(r".+@sha256:[0-9a-f]{64}", source), source
        if alias:
            stages.add(alias[-1])
    buildkit = (root / "ci/Dockerfile.buildkit").read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"FROM moby/buildkit:[^@\s]+@sha256:[0-9a-f]{64}", buildkit)
    assert "PIP_CONSTRAINT=/opt/baloot-build/requirements-docker.txt" in dockerfile
    pins = (root / "requirements-docker.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line for line in pins if line and not line.startswith("#")]
    assert requirements and all(re.fullmatch(r"[\w.-]+==[\w.+!-]+", line) for line in requirements)


def test_docker_constraint_job_resolves_the_shared_target(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    check = jobs["docker-constraints"]
    assert check["needs"] == "plan"
    assert check["if"] == "needs.plan.outputs.docker_constraints == 'true'"
    assert jobs["plan"]["outputs"]["docker_constraints"] == "${{ steps.plan.outputs.docker_constraints }}"
    assert check["steps"][-1]["run"] == "docker build --progress=plain --target docker-constraints ."
    assert "docker-constraints" not in jobs["media-inputs"]["needs"]


def test_address_sanitizer_requires_both_verifier_nodes_against_its_build(root):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    sanitizer = jobs["native-sanitizers"]
    assert "native-sanitizers" in jobs["ci-gate"]["needs"]
    build, verify = sanitizer["steps"][-2:]
    assert 'cmake --build "$SANITIZER_BUILD_DIR"' in build["run"]
    assert verify["if"] == "matrix.sanitizer == 'address'"
    assert "continue-on-error" not in verify
    assert 'BALOOT_ASAN_BUILD_DIR="$SANITIZER_BUILD_DIR" python -m scripts.run_required_pytest' in verify["run"]
    assert re.findall(r"tests/\S+\.py::\w+", verify["run"]) == [
        "tests/test_v20_asan_verifier_portability.py::test_generated_asan_ctest_invokes_sanitized_binary_without_preload",
        "tests/test_v20_asan_verifier_portability.py::test_configured_sanitized_binary_passes_without_preload",
    ]


def test_configured_asan_build_rejects_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("BALOOT_ASAN_BUILD_DIR", str(tmp_path))
    for filename in ("CTestTestfile.cmake", "baloot_native_tests"):
        with pytest.raises(pytest.fail.Exception, match="configured ASan build is missing"):
            asan_portability._asan_artifact(filename, "normal release verifier skip")
        path = tmp_path / filename
        path.touch()
        assert asan_portability._asan_artifact(filename, "normal release verifier skip") == path


def test_windows_job_requires_all_platform_nodes_and_the_capped_launcher(root, monkeypatch):
    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    windows = jobs["python-windows"]
    assert windows["runs-on"] == "windows-latest"
    assert windows["if"] == "needs.plan.outputs.full == 'true'"
    assert "python-windows" in jobs["ci-gate"]["needs"] and "python-windows" in ci_scope.FULL
    assert "python-windows" in jobs["python-backend-run"]["needs"]
    configure, verify, capped = windows["steps"][-3:]
    assert configure["run"] == "Copy-Item -LiteralPath ci/windows-ci-local-cpu.json -Destination configs/runtime/local_cpu.json"
    assert json.loads((root / "ci/windows-ci-local-cpu.json").read_bytes()) == {"schemaVersion": 1, "cpuPercent": 25}
    assert "python -m scripts.run_required_pytest @nodes" in verify["run"]
    assert "./scripts/run_local_capped.ps1" in capped["run"]
    assert "'scripts.run_required_pytest', $env:CAPPED_PYTEST_NODE" in capped["run"]
    assert capped["if"] == f"!cancelled() && steps.{configure['id']}.outcome == 'success'"
    assert all("continue-on-error" not in step for step in (configure, verify, capped))
    nodes = verify["env"]["WINDOWS_PYTEST_NODES"].splitlines() + [capped["env"]["CAPPED_PYTEST_NODE"]]
    assert len(nodes) == len(set(nodes)) == 12
    assert nodes[-1] == "tests/test_local_cpu_launcher.py::test_current_capped_process_has_verified_inherited_limits"
    paths = {node.split("::")[0] for node in nodes}
    platform_tests = {
        f"{path}::{test.name}"
        for path in paths for test in ast.parse((root / path).read_bytes()).body
        if isinstance(test, ast.FunctionDef)
        and any(ast.unparse(mark).startswith("pytest.mark.skipif(os.name != 'nt',") for mark in test.decorator_list)
    }
    assert {node.split("[")[0] for node in nodes} == platform_tests
    for path in paths:
        monkeypatch.setattr(ci_scope, "changed_paths", lambda *_, path=path: {path})
        assert ci_scope.current_plan("pull_request")["full"]


@pytest.mark.parametrize("outcome", ["passed", "skipped"])
def test_required_pytest_uses_exact_nodes_and_rejects_skips(monkeypatch, outcome):
    node = "tests/test_sample.py::test_example"
    monkeypatch.setattr(run_required_pytest.sys, "argv", ["run_required_pytest", node])

    def run(command, *, env, check):
        assert command[-1] == node and check is False
        assert command[command.index("-p") + 1] == "scripts.run_heavy_pytest"
        receipt = Path(env["BALOOT_CI_NODE_RECEIPT"])
        receipt.write_text(json.dumps({
            "exitstatus": 0, "deselected": [], "collectionIssues": [], "collected": [node],
            "reports": {node: [{"phase": phase, "outcome": outcome, "xfail": False}
                               for phase in ("setup", "call", "teardown")]},
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_required_pytest.subprocess, "run", run)
    if outcome == "passed":
        assert run_required_pytest.main() == 0
    else:
        with pytest.raises(ValueError, match="without skipping"):
            run_required_pytest.main()


def test_viewer_seat_proof_dispatch_and_nightly_use_the_same_pinned_cohort(root):
    import hashlib

    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())
    option = workflow[True]["workflow_dispatch"]["inputs"]["viewer-seat-proof"]
    assert option["type"] == "boolean" and option["default"] is False
    jobs = workflow["jobs"]
    assert jobs["preflight"]["if"] == "${{ !inputs.viewer-seat-proof }}"
    workflow_call = {"uses": "./.github/workflows/viewer-seat-proof.yml"}
    assert jobs["viewer-seat-proof"] == {
        **workflow_call,
        "if": "${{ github.event_name == 'workflow_dispatch' && inputs.viewer-seat-proof }}",
    }
    assert {name for name, job in jobs.items() if "needs" not in job} == {
        "plan", "preflight", "dependency-review", "viewer-seat-proof",
    }
    nightly = yaml.safe_load((root / ".github/workflows/nightly-candidates.yml").read_bytes())
    assert nightly["jobs"]["viewer-seat-proof"] == workflow_call
    shared = yaml.safe_load((root / workflow_call["uses"]).read_bytes())
    assert shared[True] == {"workflow_call": None}
    assert shared["permissions"] == {"contents": "read"}
    assert set(shared["jobs"]) == {"proof"}
    proof = shared["jobs"]["proof"]
    assert not {"if", "needs", "permissions", "continue-on-error"} & proof.keys()
    assert proof["runs-on"] == "ubuntu-24.04" and proof["timeout-minutes"] == 15
    assert proof["strategy"]["fail-fast"] is False
    assert set(proof["strategy"]["matrix"]) == {"include"}
    cohort = proof["strategy"]["matrix"]["include"]
    assert len(cohort) == 3
    assert {row["clip"] for row in cohort} == {"16-05", "19-03", "18-03"}
    for row in cohort:
        path = root / f"examples/research-scenario-v4-single-winner-202611{row['clip']}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert proof["steps"][0]["with"] == {"persist-credentials": False}
    for step in proof["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
        assert "continue-on-error" not in step
    run, upload = proof["steps"][-2:]
    assert "python -P scripts/viewer_seat_proof.py" in run["run"]
    assert run["env"]["FIXTURE_SHA256"] == "${{ matrix.sha256 }}"
    assert '--fixture-sha256 "$FIXTURE_SHA256"' in run["run"]
    assert upload["if"] == "${{ !cancelled() && steps.proof.outcome != 'skipped' }}"
    assert upload["with"] == {
        "name": "viewer-seat-${{ matrix.clip }}",
        "path": "viewer-seat-${{ matrix.clip }}.json",
        "if-no-files-found": "error",
        "retention-days": 30,
    }


def test_nightly_candidates_serializes_runs_and_retains_failure_evidence(root):
    workflow = yaml.safe_load((root / ".github/workflows/nightly-candidates.yml").read_bytes())
    assert workflow[True] == {"schedule": [{"cron": "17 0 * * *"}], "workflow_dispatch": None}
    assert workflow["concurrency"] == {"group": "nightly-candidates", "cancel-in-progress": False}
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["nightly-candidates"]
    seal = workflow["jobs"]["seal-pool"]
    assert job["strategy"]["matrix"] == {"shard": [0, 1, 2, 3]}
    assert seal["needs"] == "nightly-candidates"
    assert seal["runs-on"] == "ubuntu-24.04"
    assert seal["steps"][-1]["with"]["retention-days"] == 3
    assert "sha256sum --check --strict SHA256SUMS.txt" in seal["steps"][-2]["run"]
    assert "secrets." not in json.dumps(workflow["jobs"])
    assert job["runs-on"] == "ubuntu-24.04" and job["timeout-minutes"] == 25
    assert "permissions" not in job
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
    assert job["steps"][1]["with"]["persist-credentials"] is False
    qualify, finalize, upload = job["steps"][-3:]
    assert sum(step["timeout-minutes"] for step in job["steps"][:-3] + [finalize, upload]) < job["timeout-minutes"]
    assert qualify["id"] == "qualify"
    assert '--source "$GITHUB_SHA"' in qualify["run"]
    assert 'timeout --kill-after=10s' in qualify["run"]
    assert "BALOOT_NIGHTLY_JOB_DEADLINE" in job["steps"][0]["run"]
    assert "BALOOT_NIGHTLY_JOB_DEADLINE" in qualify["run"]
    assert 'candidate-index.json' in finalize["run"]
    assert finalize["if"] == upload["if"] == "${{ !cancelled() }}"
    assert upload["with"]["path"] == "${{ runner.temp }}/nightly-candidates/"
    assert upload["with"]["if-no-files-found"] == "error"


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
        steps = job.get("steps", [])
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
    plan["docker_constraints"] = "false"
    results["docker-constraints"]["result"] = "skipped"
    if event == "pull_request":
        ci_scope.validate_results(event, results)
    else:
        with pytest.raises(ValueError, match="full CI plan must select all full producers"):
            ci_scope.validate_results(event, results)
    plan["docker_constraints"] = "true"
    results["docker-constraints"]["result"] = "success"
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
    "ci/TIMINGS.md",
    "sdk/go/go.mod", "sdk/typescript/package.json", ci_scope.FOUNDRY + "../package-lock.json",
    "tests/conftest.py", "tests/test_ci_hosted_contract.py", "tests/test_core.py",
    "tests/test_research_ortools_unified_executor.py", "tests/test_unmeasured.py",
    "docs/architecture/authority-map.json", "src/baloot_platform/core/rules.py",
    "scripts/remotion_image_parity.py", "tests/test_remotion_image_parity.py", "tests/test_v8_identity_sakka.py",
    "src/baloot_platform/video/remotion/package.json", "src/baloot_platform/video/remotion/package-lock.json",
    "LICENSE", "THIRD-PARTY-NOTICES", "native/CMakeLists.txt", "configs/runtime/docker_v1.yaml",
    "schemas/version.schema.json", "generated/openapi.json", "proto/baloot.proto", "migrations/001_initial.sql",
    "assets/fonts/manifest.json", "release/saudi-baloot-official-source-verification-20260831-v2.json",
])
def test_unknown_shared_and_ci_paths_force_the_full_profile(monkeypatch, path):
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {"README.md", path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan() | {"docker_constraints": False}


@pytest.mark.parametrize("path", [
    "Dockerfile", ".dockerignore", "Dockerfile.dockerignore", "pyproject.toml", "requirements-docker.txt",
    ".github/workflows/ci.yml", "scripts/ci_scope.py",
])
def test_docker_constraint_inputs_select_the_resolver(monkeypatch, path):
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan()


def test_plan_falls_back_for_documentation_anchor(root, monkeypatch):
    profile = yaml.safe_load((root / "configs/adjudication/research_events_v1.yaml").read_bytes())
    path = profile["events"]["out_of_turn"]["documentation"].split("#")[0]
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {path})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan() | {"docker_constraints": False}


def test_plan_falls_back_for_cross_test_path(root, monkeypatch):
    acceptance = (root / "tests/test_research_ortools_authority_acceptance.py").read_bytes()
    match = re.search(rb'"(tests/[^"]+\.py)::test_preference_miss_does_not_reject', acceptance)
    assert match
    monkeypatch.setattr(ci_scope, "changed_paths", lambda *_: {match[1].decode()})
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan() | {"docker_constraints": False}


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
    assert ci_scope.current_plan("pull_request") == ci_scope.current_plan() | {"docker_constraints": False}


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
    assert affected["steps"][2]["with"]["python-version"] == "${{ matrix.python }}"
    assert any(step.get("env", {}).get("AFFECTED_TESTS") == "${{ needs.plan.outputs.tests }}"
               for step in affected["steps"])
    assert "python-affected" in jobs["ci-gate"]["needs"]


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_file_receipts_reject_unexpected_skip_xfail_and_xpass(phase):
    from scripts import run_isolated_pytest as isolated

    node = "tests/test_sample.py::test_case"
    phases = ["setup", "teardown"] if phase == "setup" else ["setup", "call", "teardown"]
    events = [{"phase": when, "outcome": "passed", "xfail": False} for when in phases]
    event = next(row for row in events if row["phase"] == phase)
    receipt = {"exitstatus": 0, "deselected": [], "collectionIssues": [], "collected": [node],
               "reports": {node: events}}
    event.update(outcome="skipped", skipReason="unexpected dependency failure")
    with pytest.raises(ValueError, match=f"unexpected pytest skip in {phase}"):
        isolated.validate_execution(receipt, {node: (phase, "requires Windows")})
    allowed = {node: (phase, event["skipReason"])}
    if phase == "teardown":
        with pytest.raises(ValueError, match="unexpected pytest skip in teardown"):
            isolated.validate_execution(receipt, allowed)
    else:
        isolated.validate_execution(receipt, allowed)
    event["xfail"] = True
    for outcome in ("skipped", "passed"):
        event["outcome"] = outcome
        if phase == "setup" and outcome == "passed":
            events.insert(1, {"phase": "call", "outcome": "passed", "xfail": False})
        with pytest.raises(ValueError, match="unexpected pytest xfail/xpass"):
            isolated.validate_execution(receipt, allowed)
    receipt["collectionIssues"] = [{"nodeid": "tests/test_sample.py", "outcome": "skipped"}]
    with pytest.raises(ValueError, match="skipped/failed collection"):
        isolated.validate_execution(receipt, allowed)


def test_python_job_deadline_and_ortools_wheelhouse_follow_the_matrix(root, monkeypatch):
    from scripts import run_isolated_pytest as isolated

    jobs = yaml.safe_load((root / ".github/workflows/ci.yml").read_bytes())["jobs"]
    deadline = jobs["python-fast-run"]["steps"][0]
    assert '$(date +%s) + BALOOT_PYTEST_JOB_MINUTES * 60 - 60' in deadline["run"]
    assert '>> "$GITHUB_ENV"' in deadline["run"]
    for name in ("python-fast-run", "python-backend-run", "python-heavy", "python-affected"):
        job = jobs[name]
        assert job["steps"][0] is deadline
        assert job["env"]["BALOOT_PYTEST_JOB_MINUTES"] == job["timeout-minutes"]
    monkeypatch.setattr(isolated.time, "time", lambda: 1000)
    for cutoff, remaining in ((2320, 1320), (1050, 50), (980, 0)):
        monkeypatch.setenv("BALOOT_PYTEST_DEADLINE", str(cutoff))
        assert isolated.job_seconds_remaining(900) == remaining
    wheelhouse, heavy = jobs["python-wheelhouse"], jobs["python-heavy"]
    assert "python-wheelhouse" in heavy["needs"]
    assert heavy["strategy"]["matrix"]["python"] == wheelhouse["strategy"]["matrix"]["python"]
    install, verify = [next(step for step in wheelhouse["steps"] if step.get("name") == name)
                       for name in ("Build wheels and install the run-local Python runtime",
                                    "Verify OR-Tools in the installed runtime")]
    assert "if" not in install and "if" not in verify
    assert wheelhouse["steps"].index(verify) == wheelhouse["steps"].index(install) + 1
    assert '"$RUNNER_TEMP/ci-venv/bin/python" -I -c' in verify["run"]
    assert "from ortools.sat.python import cp_model; cp_model.CpModel()" in verify["run"]
    assert any(step.get("with", {}).get("name") == "python-runtime-${{ matrix.python }}" for step in heavy["steps"])


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


def test_dependabot_groups_weekly_version_minor_and_patch_updates(root):
    updates = yaml.safe_load((root / ".github/dependabot.yml").read_bytes())["updates"]
    assert {row["package-ecosystem"] for row in updates} == {"github-actions", "pip", "npm", "gomod", "docker"}
    for update in updates:
        assert update["schedule"]["interval"] == "weekly"
        group, = update["groups"].values()
        assert group == {"patterns": ["*"], "update-types": ["minor", "patch"]}

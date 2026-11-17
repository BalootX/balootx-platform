import json
import os
import subprocess
import sys


def test_openapi_contract_matches_application(root):
    from baloot_platform.service.api import app

    stored = json.loads((root / "api/openapi.json").read_text(encoding="utf-8"))
    assert stored == app.openapi()


def test_ci_fails_when_openapi_generation_changes_tracked_contract(root):
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "git diff --exit-code -- api/openapi.json" in workflow


def test_openapi_check_is_clean_source_checkout_safe(root):
    result = subprocess.run(
        [sys.executable, "-B", "scripts/generate_openapi.py", "--check"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        # Alwaleed green p99 is 41.3s; 3x margin rounded up to 180s.
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr

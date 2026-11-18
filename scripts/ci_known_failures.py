"""Exact first-run resource failures; unlisted nodes or changed causes stay red."""
from functools import lru_cache
import json
import os
import platform
import subprocess
from pathlib import Path
import re

BASELINE = Path(__file__).resolve().parents[1] / "ci/known-failures.json"


def failure_signature(longrepr: str) -> dict | None:
    errors = re.findall(r"(?m)^E\s+(.+)$", longrepr)
    if not errors:
        return None
    message = errors[-1]
    match = re.fullmatch(r"FileNotFoundError: \[Errno 2\] No such file or directory: ['\"](.+?)['\"]", message)
    if match:
        resource = match[1].replace("\\", "/")
        for prefix in ("assets/", "outputs/"):
            offset = resource.find(prefix)
            if offset >= 0:
                return {"exception": "FileNotFoundError", "resourceKind": "missing_file",
                        "resource": resource[offset:]}
    return None


@lru_cache(maxsize=1)
def entries() -> dict:
    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name in ("pull_request", "merge_group"):
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
        base = (event["pull_request"]["base"]["sha"] if event_name == "pull_request"
                else event["merge_group"]["base_sha"])
        if not re.fullmatch(r"[0-9a-f]{40}", base):
            raise ValueError("invalid trusted base SHA")
        try:
            result = subprocess.run(["git", "show", f"{base}:ci/known-failures.json"],
                cwd=BASELINE.parents[1], capture_output=True, text=True, check=True, timeout=10)
        except subprocess.CalledProcessError as exc:
            raise ValueError("known-failure baseline is absent from the trusted base; explicit Portfolio bootstrap admission is required") from exc
        baseline = json.loads(result.stdout)
    elif event_name == "push" and os.environ.get("GITHUB_REF") == "refs/heads/main":
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    else:
        return {}
    if baseline["schemaVersion"] != 1 or baseline["source"]["runId"] != 44039515022:
        raise ValueError("unsupported known-failure baseline")
    result = {}
    version = ".".join(platform.python_version_tuple()[:2])
    for row in baseline["tests"]:
        if not any(f"-{version}-" in observation["artifactName"] for observation in row["observations"]):
            continue
        result.setdefault((row["nodeid"], row["phase"]), []).append(row["signature"])
    return result


def is_known(nodeid: str, event: dict) -> bool:
    if event["outcome"] != "failed" or event.get("xfail"):
        return False
    signature = failure_signature(event.get("longrepr", ""))
    return signature is not None and signature in entries().get((nodeid, event["phase"]), ())

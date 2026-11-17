#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from baloot_platform.service.api import app
from baloot_platform.service.mcp_server import TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]


def grpc_operations() -> set[str]:
    values: set[str] = set()
    for path in sorted((ROOT / "proto/baloot").rglob("*.proto")):
        text = path.read_text(encoding="utf-8")
        package_match = re.search(r"^package\s+([A-Za-z0-9_.]+)\s*;", text, re.M)
        if not package_match:
            continue
        package = package_match.group(1)
        for service_match in re.finditer(r"service\s+(\w+)\s*\{(.*?)\}", text, re.S):
            service, body = service_match.groups()
            for method in re.findall(r"\brpc\s+(\w+)\s*\(", body):
                values.add(f"{package}.{service}/{method}")
    return values


def main() -> None:
    matrix = yaml.safe_load((ROOT / "configs/capabilities/platform_v19.yaml").read_text(encoding="utf-8"))
    schema = app.openapi()
    rest = {
        f"{method.upper()} {path}"
        for path, value in schema.get("paths", {}).items()
        for method in value
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }
    mcp = {str(item["name"]) for item in TOOL_DEFINITIONS}
    grpc = grpc_operations()
    results: dict[str, dict[str, object]] = {}
    passed = True
    for name, contract in sorted(dict(matrix.get("capabilities") or {}).items()):
        expected_rest = str(contract.get("rest") or "")
        expected_mcp = str(contract.get("mcp") or "")
        expected_grpc = str(contract.get("grpc") or "")
        missing = {
            "rest": [] if expected_rest in rest else [expected_rest],
            "mcp": [] if expected_mcp in mcp else [expected_mcp],
            "grpc": [] if expected_grpc in grpc else [expected_grpc],
        }
        item_passed = not any(missing.values())
        passed &= item_passed
        results[name] = {"passed": item_passed, "contract": contract, "missing": missing}
    payload = {
        "schemaVersion": 2,
        "version": app.version,
        "status": "passed" if passed else "failed",
        "declaredCapabilities": len(results),
        "restOperations": len(rest),
        "mcpTools": len(mcp),
        "grpcOperations": len(grpc),
        "capabilities": results,
    }
    payload["manifestHash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = ROOT / "generated/capability-parity-v19.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if not passed:
        failed = [name for name, value in results.items() if not value["passed"]]
        raise SystemExit(f"capability parity failed: {failed}")


if __name__ == "__main__":
    main()

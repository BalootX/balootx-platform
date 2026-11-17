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


def sdk_operations() -> dict[str, set[str]]:
    python_source = (ROOT / "sdk/python/client.py").read_text(encoding="utf-8")
    typescript_source = (ROOT / "sdk/typescript/src/client.ts").read_text(encoding="utf-8")
    go_source = (ROOT / "sdk/go/client.go").read_text(encoding="utf-8")
    return {
        "python": set(re.findall(r"^\s+def\s+(\w+)\s*\(", python_source, re.M)),
        "typescript": set(re.findall(r"^\s+(\w+)\s*\(", typescript_source, re.M)),
        "go": set(re.findall(r"^func\s+\(c\s+\*Client\)\s+(\w+)\s*\(", go_source, re.M)),
    }


def main() -> None:
    matrix = yaml.safe_load((ROOT / "configs/capabilities/platform_v20.yaml").read_text(encoding="utf-8"))
    schema = app.openapi()
    rest = {f"{method.upper()} {path}" for path, value in schema.get("paths", {}).items() for method in value if method.lower() in {"get","post","put","patch","delete"}}
    mcp = {str(item["name"]): item for item in TOOL_DEFINITIONS}
    grpc = grpc_operations()
    sdk = sdk_operations()
    required_fields = set(matrix["requirements"]["semantic_fields_required"])
    results = {}
    passed = True
    for name, contract in sorted(matrix["operations"].items()):
        expected_rest = f"{contract['rest']['method']} {contract['rest']['path']}"
        missing = {
            "rest": [] if expected_rest in rest else [expected_rest],
            "mcp": [] if contract["mcp"] in mcp else [contract["mcp"]],
            "grpc": [] if contract["grpc"] in grpc else [contract["grpc"]],
            "sdk": [
                f"{language}:{method}"
                for language, method in sorted(contract["sdk"].items())
                if method not in sdk.get(language, set())
            ],
            "semantics": sorted(required_fields - set(contract)),
            "mcpAnnotations": [] if contract["mcp"] not in mcp or "annotations" in mcp[contract["mcp"]] else [contract["mcp"]],
        }
        item_passed = not any(missing.values())
        passed &= item_passed
        semantic_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        results[name] = {"passed": item_passed, "contract": contract, "semanticHash": semantic_hash, "missing": missing}
    payload = {"schemaVersion": 3, "version": app.version, "status": "passed" if passed else "failed",
               "declaredOperations": len(results), "restOperations": len(rest), "mcpTools": len(mcp), "grpcOperations": len(grpc),
               "semanticParity": True, "operations": results}
    payload["manifestHash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    output = ROOT / "generated/capability-parity-v20.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(output)
    if not passed:
        raise SystemExit("V20 capability parity failed")

if __name__ == "__main__":
    main()

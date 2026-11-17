"""Generate protobuf and gRPC Python bindings without requiring grpcio-tools.

The repository's typed contract is the single source of truth.  We ask any
available ``protoc`` binary for a descriptor set, then emit modern
``descriptor_pool.AddSerializedFile`` bindings that work with current protobuf
runtimes.  The small gRPC adapter is generated directly from the service
methods and intentionally contains no Baloot logic.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from importlib.util import find_spec
from pathlib import Path

from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto/baloot/v2/platform.proto"
OUT = ROOT / "src/baloot_platform/generated"


def protoc_command() -> list[str]:
    candidates = [os.environ.get("PROTOC"), shutil.which("protoc")]
    try:
        import torch  # type: ignore

        candidates.append(str(Path(torch.__file__).resolve().parent / "bin/protoc"))
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return [str(Path(candidate))]
    if find_spec("grpc_tools.protoc") is not None:
        return [sys.executable, "-m", "grpc_tools.protoc"]
    raise RuntimeError("protoc not found; install protobuf-compiler or set PROTOC")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "__init__.py").touch()
    protoc = protoc_command()
    with tempfile.TemporaryDirectory(prefix="baloot-proto-") as temp:
        descriptor_path = Path(temp) / "platform.desc"
        subprocess.run(
            [
                *protoc,
                f"-I{ROOT / 'proto'}",
                f"--descriptor_set_out={descriptor_path}",
                "--include_imports",
                str(PROTO),
            ],
            check=True,
        )
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor_path.read_bytes())
    file_proto = next((item for item in descriptor_set.file if item.name == "baloot/v2/platform.proto"), None)
    if file_proto is None:
        raise RuntimeError("platform.proto missing from descriptor set")

    pb2 = f'''# Generated from proto/baloot/v2/platform.proto by scripts/generate_grpc.py.\n# Do not edit by hand.\nfrom google.protobuf import descriptor_pool as _descriptor_pool\nfrom google.protobuf.internal import builder as _builder\n\nDESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({file_proto.SerializeToString()!r})\n_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n'''
    (OUT / "platform_pb2.py").write_text(pb2, encoding="utf-8")

    service = next((item for item in file_proto.service if item.name == "BalootPlatform"), None)
    if service is None:
        raise RuntimeError("BalootPlatform service missing")
    specs = []
    for method in service.method:
        request = method.input_type.rsplit(".", 1)[-1]
        response = method.output_type.rsplit(".", 1)[-1]
        specs.append(f'    {method.name!r}: (platform__pb2.{request}, platform__pb2.{response}),')
    grpc_code = '''# Generated-compatible gRPC bindings for baloot.v2.BalootPlatform.\n# Do not edit by hand.\nimport grpc\nfrom . import platform_pb2 as platform__pb2\n\n\n_SPECS = {\n''' + "\n".join(specs) + '''\n}\n\n\nclass BalootPlatformStub:\n    def __init__(self, channel):\n        for name, (request_type, response_type) in _SPECS.items():\n            setattr(self, name, channel.unary_unary(\n                f"/baloot.v2.BalootPlatform/{name}",\n                request_serializer=request_type.SerializeToString,\n                response_deserializer=response_type.FromString,\n                _registered_method=True,\n            ))\n\n\nclass BalootPlatformServicer:\n    pass\n\n\ndef add_BalootPlatformServicer_to_server(servicer, server):\n    handlers = {}\n    for name, (request_type, response_type) in _SPECS.items():\n        method = getattr(servicer, name)\n        handlers[name] = grpc.unary_unary_rpc_method_handler(\n            method,\n            request_deserializer=request_type.FromString,\n            response_serializer=response_type.SerializeToString,\n        )\n    generic = grpc.method_handlers_generic_handler("baloot.v2.BalootPlatform", handlers)\n    server.add_generic_rpc_handlers((generic,))\n    if hasattr(server, "add_registered_method_handlers"):\n        server.add_registered_method_handlers("baloot.v2.BalootPlatform", handlers)\n'''
    (OUT / "platform_pb2_grpc.py").write_text(grpc_code, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

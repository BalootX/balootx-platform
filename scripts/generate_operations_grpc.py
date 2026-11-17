from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "proto/baloot/v17/operations.proto"
OUT = ROOT / "src/baloot_platform/generated"


def find_protoc() -> str:
    candidates = [os.environ.get("PROTOC"), shutil.which("protoc")]
    try:
        import torch  # type: ignore
        candidates.append(str(Path(torch.__file__).resolve().parent / "bin/protoc"))
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    raise RuntimeError("protoc not found")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "__init__.py").touch()
    with tempfile.TemporaryDirectory(prefix="baloot-v17-proto-") as temp:
        descriptor = Path(temp) / "operations.desc"
        subprocess.run([
            find_protoc(), f"-I{ROOT / 'proto'}", f"--descriptor_set_out={descriptor}",
            "--include_imports", str(PROTO),
        ], check=True)
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes())
    file_proto = next(item for item in descriptor_set.file if item.name == "baloot/v17/operations.proto")
    pb2 = f'''# Generated from proto/baloot/v17/operations.proto.\nfrom google.protobuf import descriptor_pool as _descriptor_pool\nfrom google.protobuf.internal import builder as _builder\n\nDESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({file_proto.SerializeToString()!r})\n_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n'''
    (OUT / "operations_pb2.py").write_text(pb2, encoding="utf-8")
    service = next(item for item in file_proto.service if item.name == "BalootOperations")
    specs=[]
    for method in service.method:
        req=method.input_type.rsplit('.',1)[-1]; resp=method.output_type.rsplit('.',1)[-1]
        specs.append(f'    {method.name!r}: (operations__pb2.{req}, operations__pb2.{resp}),')
    grpc_code='''# Generated-compatible gRPC bindings for baloot.v17.BalootOperations.\nimport grpc\nfrom . import operations_pb2 as operations__pb2\n\n_SPECS = {\n''' + "\n".join(specs) + '''\n}\n\nclass BalootOperationsStub:\n    def __init__(self, channel):\n        for name,(request_type,response_type) in _SPECS.items():\n            setattr(self,name,channel.unary_unary(\n                f"/baloot.v17.BalootOperations/{name}",\n                request_serializer=request_type.SerializeToString,\n                response_deserializer=response_type.FromString,\n                _registered_method=True,\n            ))\n\nclass BalootOperationsServicer:\n    pass\n\ndef add_BalootOperationsServicer_to_server(servicer,server):\n    handlers={}\n    for name,(request_type,response_type) in _SPECS.items():\n        handlers[name]=grpc.unary_unary_rpc_method_handler(\n            getattr(servicer,name),\n            request_deserializer=request_type.FromString,\n            response_serializer=response_type.SerializeToString,\n        )\n    generic=grpc.method_handlers_generic_handler("baloot.v17.BalootOperations",handlers)\n    server.add_generic_rpc_handlers((generic,))\n    if hasattr(server,"add_registered_method_handlers"):\n        server.add_registered_method_handlers("baloot.v17.BalootOperations",handlers)\n'''
    (OUT / "operations_pb2_grpc.py").write_text(grpc_code, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

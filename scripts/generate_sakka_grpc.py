#!/usr/bin/env python3
"""Generate the small Sakka protobuf module without requiring protoc.

The descriptor is constructed from the checked-in proto contract so clean
builds only require the protobuf runtime. The generated gRPC bindings follow
exactly the same stable pattern as the main platform bindings.
"""
from __future__ import annotations

from pathlib import Path
from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/baloot_platform/generated"
OUT.mkdir(parents=True, exist_ok=True)

fd = descriptor_pb2.FileDescriptorProto()
fd.name = "baloot/v8/sakka.proto"
fd.package = "baloot.v8"
fd.syntax = "proto3"

TYPE = descriptor_pb2.FieldDescriptorProto

def message(name: str, fields: list[tuple[str, int, int]]) -> None:
    msg = fd.message_type.add(); msg.name = name
    for field_name, number, type_id in fields:
        field = msg.field.add(); field.name = field_name; field.number = number
        field.label = TYPE.LABEL_OPTIONAL; field.type = type_id

message("Empty", [])
message("Identifier", [("value", 1, TYPE.TYPE_STRING)])
message("JsonPayload", [("canonical_json", 1, TYPE.TYPE_BYTES)])
message("ListBoardsRequest", [("limit", 1, TYPE.TYPE_UINT32)])
message("RunActionRequest", [("run_id", 1, TYPE.TYPE_STRING), ("canonical_json", 2, TYPE.TYPE_BYTES)])
message("RunAuditRequest", [("run_id", 1, TYPE.TYPE_STRING), ("deep", 2, TYPE.TYPE_BOOL), ("force", 3, TYPE.TYPE_BOOL), ("maximum_decisions", 4, TYPE.TYPE_UINT32)])
message("ShareDecisionRequest", [("run_id", 1, TYPE.TYPE_STRING), ("sequence", 2, TYPE.TYPE_UINT32), ("canonical_json", 3, TYPE.TYPE_BYTES)])
message("FeedbackRequest", [("run_id", 1, TYPE.TYPE_STRING), ("sequence", 2, TYPE.TYPE_UINT32), ("canonical_json", 3, TYPE.TYPE_BYTES)])
message("JsonResult", [("canonical_json", 1, TYPE.TYPE_BYTES)])

svc = fd.service.add(); svc.name = "SakkaPlatform"
methods = [
    ("ListBoards", "ListBoardsRequest"), ("CreateBoard", "JsonPayload"),
    ("GetBoard", "Identifier"), ("VerifyBoard", "Identifier"), ("CreateRoom", "JsonPayload"),
    ("GetRoom", "Identifier"), ("JoinRoom", "Identifier"),
    ("StartRun", "Identifier"), ("GetRun", "Identifier"), ("VerifyRun", "Identifier"),
    ("PlayAction", "RunActionRequest"), ("AuditRun", "RunAuditRequest"),
    ("ShareDecision", "ShareDecisionRequest"), ("RecordFeedback", "FeedbackRequest"),
    ("GetLeaderboard", "Identifier"),
]
for name, input_name in methods:
    method = svc.method.add(); method.name = name
    method.input_type = f".baloot.v8.{input_name}"
    method.output_type = ".baloot.v8.JsonResult"

raw = fd.SerializeToString()
(OUT / "sakka_pb2.py").write_text(
    "# Generated from proto/baloot/v8/sakka.proto by scripts/generate_sakka_grpc.py.\n"
    "# Do not edit by hand.\n"
    "from google.protobuf import descriptor_pool as _descriptor_pool\n"
    "from google.protobuf.internal import builder as _builder\n\n"
    f"DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({raw!r})\n"
    "_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n"
    "_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n",
    encoding="utf-8",
)

spec_lines = [
    "# Generated-compatible gRPC bindings for baloot.v8.SakkaPlatform.",
    "# Do not edit by hand.",
    "import grpc",
    "from . import sakka_pb2 as sakka__pb2",
    "",
    "_SPECS = {",
]
for name, input_name in methods:
    spec_lines.append(f"    {name!r}: (sakka__pb2.{input_name}, sakka__pb2.JsonResult),")
spec_lines.extend([
    "}", "", "class SakkaPlatformStub:", "    def __init__(self, channel):",
    "        for name, (request_type, response_type) in _SPECS.items():",
    "            setattr(self, name, channel.unary_unary(",
    "                f'/baloot.v8.SakkaPlatform/{name}',",
    "                request_serializer=request_type.SerializeToString,",
    "                response_deserializer=response_type.FromString,",
    "                _registered_method=True,",
    "            ))", "", "class SakkaPlatformServicer:", "    pass", "",
    "def add_SakkaPlatformServicer_to_server(servicer, server):",
    "    handlers = {}",
    "    for name, (request_type, response_type) in _SPECS.items():",
    "        method = getattr(servicer, name)",
    "        handlers[name] = grpc.unary_unary_rpc_method_handler(",
    "            method, request_deserializer=request_type.FromString,",
    "            response_serializer=response_type.SerializeToString,",
    "        )",
    "    generic = grpc.method_handlers_generic_handler('baloot.v8.SakkaPlatform', handlers)",
    "    server.add_generic_rpc_handlers((generic,))",
    "    if hasattr(server, 'add_registered_method_handlers'):",
    "        server.add_registered_method_handlers('baloot.v8.SakkaPlatform', handlers)",
])
(OUT / "sakka_pb2_grpc.py").write_text("\n".join(spec_lines) + "\n", encoding="utf-8")
print(OUT / "sakka_pb2.py")

#!/usr/bin/env python3
"""Generate V9 shared-play protobuf modules without requiring protoc."""
from __future__ import annotations
from pathlib import Path
from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/baloot_platform/generated"
OUT.mkdir(parents=True, exist_ok=True)
fd = descriptor_pb2.FileDescriptorProto()
fd.name = "baloot/v9/shared.proto"; fd.package = "baloot.v9"; fd.syntax = "proto3"
TYPE = descriptor_pb2.FieldDescriptorProto

def message(name: str, fields: list[tuple[str,int,int]]) -> None:
    msg=fd.message_type.add(); msg.name=name
    for field_name, number, type_id in fields:
        field=msg.field.add(); field.name=field_name; field.number=number
        field.label=TYPE.LABEL_OPTIONAL; field.type=type_id

message("Identifier", [("value",1,TYPE.TYPE_STRING)])
message("JsonPayload", [("canonical_json",1,TYPE.TYPE_BYTES)])
message("ListBoardsRequest", [("limit",1,TYPE.TYPE_UINT32)])
message("RunActionRequest", [("run_id",1,TYPE.TYPE_STRING),("canonical_json",2,TYPE.TYPE_BYTES)])
message("RunReviewRequest", [("run_id",1,TYPE.TYPE_STRING),("ensure_audit",2,TYPE.TYPE_BOOL)])
message("DecisionRequest", [("run_id",1,TYPE.TYPE_STRING),("decision_id",2,TYPE.TYPE_STRING)])
message("ExplainDecisionRequest", [
    ("run_id",1,TYPE.TYPE_STRING),("decision_id",2,TYPE.TYPE_STRING),("level",3,TYPE.TYPE_STRING),
    ("surface",4,TYPE.TYPE_STRING),("challenge_revealed",5,TYPE.TYPE_BOOL),("ensure_audit",6,TYPE.TYPE_BOOL),
    ("view_mode",7,TYPE.TYPE_STRING),
])
message("ReviewDecisionRequest", [("run_id",1,TYPE.TYPE_STRING),("decision_id",2,TYPE.TYPE_STRING),("canonical_json",3,TYPE.TYPE_BYTES)])
message("ShareDecisionRequest", [("run_id",1,TYPE.TYPE_STRING),("sequence",2,TYPE.TYPE_UINT32),("canonical_json",3,TYPE.TYPE_BYTES)])
message("CompareRunsRequest", [("run_a",1,TYPE.TYPE_STRING),("run_b",2,TYPE.TYPE_STRING)])
message("JsonResult", [("canonical_json",1,TYPE.TYPE_BYTES)])
svc=fd.service.add(); svc.name="SharedPlatform"
methods=[
    ("ListBoards","ListBoardsRequest"),("CreateBoard","JsonPayload"),("GetBoard","Identifier"),("VerifyBoard","Identifier"),
    ("CreateRoom","JsonPayload"),("GetRoom","Identifier"),("JoinRoom","Identifier"),("StartRun","Identifier"),
    ("GetRun","Identifier"),("VerifyRun","Identifier"),("PlayAction","RunActionRequest"),
    ("GetRunReview","RunReviewRequest"),("ExplainDecision","ExplainDecisionRequest"),
    ("GetDecisionReviews","DecisionRequest"),("SubmitDecisionReview","ReviewDecisionRequest"),
    ("ShareDecision","ShareDecisionRequest"),("GetLeaderboard","Identifier"),("CompareRuns","CompareRunsRequest"),
]
for name,input_name in methods:
    method=svc.method.add(); method.name=name
    method.input_type=f".baloot.v9.{input_name}"; method.output_type=".baloot.v9.JsonResult"
raw=fd.SerializeToString()
(OUT/"shared_pb2.py").write_text(
    "# Generated from proto/baloot/v9/shared.proto by scripts/generate_shared_grpc.py.\n"
    "# Do not edit by hand.\n"
    "from google.protobuf import descriptor_pool as _descriptor_pool\n"
    "from google.protobuf.internal import builder as _builder\n\n"
    f"DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({raw!r})\n"
    "_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n"
    "_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n",
    encoding="utf-8",
)
lines=[
    "# Generated-compatible gRPC bindings for baloot.v9.SharedPlatform.","# Do not edit by hand.",
    "import grpc","from . import shared_pb2 as shared__pb2","","_SPECS = {",
]
for name,input_name in methods:
    lines.append(f"    {name!r}: (shared__pb2.{input_name}, shared__pb2.JsonResult),")
lines += [
    "}","","class SharedPlatformStub:","    def __init__(self, channel):",
    "        for name, (request_type, response_type) in _SPECS.items():",
    "            setattr(self, name, channel.unary_unary(",
    "                f'/baloot.v9.SharedPlatform/{name}',",
    "                request_serializer=request_type.SerializeToString,",
    "                response_deserializer=response_type.FromString,",
    "                _registered_method=True,",
    "            ))","","class SharedPlatformServicer:","    pass","",
    "def add_SharedPlatformServicer_to_server(servicer, server):",
    "    handlers = {}","    for name, (request_type, response_type) in _SPECS.items():",
    "        method = getattr(servicer, name)",
    "        handlers[name] = grpc.unary_unary_rpc_method_handler(",
    "            method, request_deserializer=request_type.FromString,",
    "            response_serializer=response_type.SerializeToString,",
    "        )",
    "    generic = grpc.method_handlers_generic_handler('baloot.v9.SharedPlatform', handlers)",
    "    server.add_generic_rpc_handlers((generic,))",
    "    if hasattr(server, 'add_registered_method_handlers'):",
    "        server.add_registered_method_handlers('baloot.v9.SharedPlatform', handlers)",
]
(OUT/"shared_pb2_grpc.py").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(OUT/"shared_pb2.py")

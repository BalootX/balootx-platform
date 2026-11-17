#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from google.protobuf import descriptor_pb2

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'src/baloot_platform/generated'; OUT.mkdir(parents=True,exist_ok=True)
fd=descriptor_pb2.FileDescriptorProto(); fd.name='baloot/v19/configuration.proto'; fd.package='baloot.v19'; fd.syntax='proto3'; T=descriptor_pb2.FieldDescriptorProto

def msg(name,fields):
 m=fd.message_type.add(); m.name=name
 for field_name,number,type_id,type_name,label in fields:
  f=m.field.add(); f.name=field_name; f.number=number; f.type=type_id; f.label=label or T.LABEL_OPTIONAL
  if type_name: f.type_name=type_name

msg('Empty',[]); msg('Identifier',[('value',1,T.TYPE_STRING,None,None),('expected_etag',2,T.TYPE_UINT64,None,None)])
msg('TransitionRequest',[('value',1,T.TYPE_STRING,None,None),('expected_etag',2,T.TYPE_UINT64,None,None),('reason',3,T.TYPE_STRING,None,None)])
msg('ProfileQuery',[('family',1,T.TYPE_STRING,None,None),('profile_id',2,T.TYPE_STRING,None,None),('status',3,T.TYPE_STRING,None,None),('limit',4,T.TYPE_UINT32,None,None),('scope',5,T.TYPE_STRING,None,None),('tenant_id',6,T.TYPE_STRING,None,None)])
msg('ProfileDraft',[('family',1,T.TYPE_STRING,None,None),('canonical_json',2,T.TYPE_BYTES,None,None),('scope',3,T.TYPE_STRING,None,None),('tenant_id',4,T.TYPE_STRING,None,None),('supersedes_revision_id',5,T.TYPE_STRING,None,None),('publish',6,T.TYPE_BOOL,None,None)])
msg('ProfileResolution',[('family',1,T.TYPE_STRING,None,None),('profile_id',2,T.TYPE_STRING,None,None),('tenant_id',3,T.TYPE_STRING,None,None)])
msg('ProfileCompare',[('left',1,T.TYPE_STRING,None,None),('right',2,T.TYPE_STRING,None,None)])
msg('BundleQuery',[('bundle_id',1,T.TYPE_STRING,None,None),('status',2,T.TYPE_STRING,None,None),('limit',3,T.TYPE_UINT32,None,None),('scope',4,T.TYPE_STRING,None,None),('tenant_id',5,T.TYPE_STRING,None,None)])
entry=fd.message_type.add(); entry.name='BundleDraft_ComponentsEntry'; entry.options.map_entry=True
for n,name,type_id in ((1,'key',T.TYPE_STRING),(2,'value',T.TYPE_STRING)):
 f=entry.field.add(); f.name=name; f.number=n; f.type=type_id; f.label=T.LABEL_OPTIONAL
msg('BundleDraft',[('bundle_id',1,T.TYPE_STRING,None,None),('version',2,T.TYPE_STRING,None,None),('components',3,T.TYPE_MESSAGE,'.baloot.v19.BundleDraft_ComponentsEntry',T.LABEL_REPEATED),('mode',4,T.TYPE_STRING,None,None),('scope',5,T.TYPE_STRING,None,None),('tenant_id',6,T.TYPE_STRING,None,None),('supersedes_revision_id',7,T.TYPE_STRING,None,None),('publish',8,T.TYPE_BOOL,None,None),('allow_drafts',9,T.TYPE_BOOL,None,None)])
msg('BundleResolution',[('revision_id',1,T.TYPE_STRING,None,None),('bundle_id',2,T.TYPE_STRING,None,None),('tenant_id',3,T.TYPE_STRING,None,None),('allow_draft',4,T.TYPE_BOOL,None,None)])
msg('BundleRollback',[('revision_id',1,T.TYPE_STRING,None,None),('new_version',2,T.TYPE_STRING,None,None)])
msg('PluginQuery',[('kind',1,T.TYPE_STRING,None,None)])
msg('JsonPayload',[('canonical_json',1,T.TYPE_BYTES,None,None)])
msg('JsonResult',[('canonical_json',1,T.TYPE_BYTES,None,None),('sha256',2,T.TYPE_STRING,None,None)])
svc=fd.service.add(); svc.name='ConfigurationControl'
methods=[('GetCapabilities','Empty'),('ListProfiles','ProfileQuery'),('CreateProfile','ProfileDraft'),('PublishProfile','Identifier'),('RejectProfile','TransitionRequest'),('RetireProfile','TransitionRequest'),('ResolveProfile','ProfileResolution'),('CompareProfiles','ProfileCompare'),('ListBundles','BundleQuery'),('CreateBundle','BundleDraft'),('PublishBundle','Identifier'),('RejectBundle','TransitionRequest'),('RetireBundle','TransitionRequest'),('ResolveBundle','BundleResolution'),('ResolveEffectiveRuntimeConfigV2','BundleResolution'),('RollbackBundle','BundleRollback'),('ListPlugins','PluginQuery'),('GetPerformanceProfile','Empty'),('RunPerformanceQuick','Empty'),('GetCapabilityMatrix','Empty'),('GetCertificationProfile','Empty'),('VerifyCertificationCandidate','JsonPayload'),('GetBenchmarkProfile','Empty'),('EvaluateIntelligenceBenchmark','JsonPayload'),('GetOperationsProfile','Empty')]
for name,input_name in methods:
 m=svc.method.add(); m.name=name; m.input_type=f'.baloot.v19.{input_name}'; m.output_type='.baloot.v19.JsonResult'
raw=fd.SerializeToString()
(OUT/'configuration_pb2.py').write_text("# Generated from proto/baloot/v19/configuration.proto.\nfrom google.protobuf import descriptor_pool as _descriptor_pool\nfrom google.protobuf.internal import builder as _builder\n\nDESCRIPTOR = _descriptor_pool.Default().AddSerializedFile("+repr(raw)+")\n_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n",encoding='utf-8')
lines=["# Generated-compatible gRPC bindings for baloot.v19.ConfigurationControl.","import grpc","from . import configuration_pb2 as configuration__pb2","","_SPECS = {"]
for name,input_name in methods: lines.append(f"    {name!r}: (configuration__pb2.{input_name}, configuration__pb2.JsonResult),")
lines += ["}","","class ConfigurationControlStub:","    def __init__(self, channel):","        for name,(request_type,response_type) in _SPECS.items():","            setattr(self,name,channel.unary_unary(f'/baloot.v19.ConfigurationControl/{name}',request_serializer=request_type.SerializeToString,response_deserializer=response_type.FromString,_registered_method=True))","","class ConfigurationControlServicer:","    pass","","def add_ConfigurationControlServicer_to_server(servicer, server):","    handlers={}","    for name,(request_type,response_type) in _SPECS.items():","        handlers[name]=grpc.unary_unary_rpc_method_handler(getattr(servicer,name),request_deserializer=request_type.FromString,response_serializer=response_type.SerializeToString)","    generic=grpc.method_handlers_generic_handler('baloot.v19.ConfigurationControl',handlers)","    server.add_generic_rpc_handlers((generic,))","    if hasattr(server,'add_registered_method_handlers'):","        server.add_registered_method_handlers('baloot.v19.ConfigurationControl',handlers)"]
(OUT/'configuration_pb2_grpc.py').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(OUT/'configuration_pb2.py')

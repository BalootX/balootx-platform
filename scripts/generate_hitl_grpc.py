from __future__ import annotations
import os, shutil, subprocess, tempfile
from pathlib import Path
from google.protobuf import descriptor_pb2
ROOT=Path(__file__).resolve().parents[1]; PROTO=ROOT/'proto/baloot/v18/hitl.proto'; OUT=ROOT/'src/baloot_platform/generated'
def find_protoc():
 candidates=[os.environ.get('PROTOC'),shutil.which('protoc')]
 try:
  import torch
  candidates.append(str(Path(torch.__file__).resolve().parent/'bin/protoc'))
 except Exception: pass
 for candidate in candidates:
  if candidate and Path(candidate).is_file(): return str(Path(candidate))
 raise RuntimeError('protoc not found')
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'__init__.py').touch()
 with tempfile.TemporaryDirectory(prefix='baloot-v18-proto-') as temp:
  descriptor=Path(temp)/'hitl.desc'
  subprocess.run([find_protoc(),f'-I{ROOT/"proto"}',f'--descriptor_set_out={descriptor}','--include_imports',str(PROTO)],check=True)
  ds=descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes())
 fp=next(x for x in ds.file if x.name=='baloot/v18/hitl.proto')
 pb2=f'''# Generated from proto/baloot/v18/hitl.proto.\nfrom google.protobuf import descriptor_pool as _descriptor_pool\nfrom google.protobuf.internal import builder as _builder\nDESCRIPTOR = _descriptor_pool.Default().AddSerializedFile({fp.SerializeToString()!r})\n_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())\n_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())\n'''
 (OUT/'hitl_pb2.py').write_text(pb2,encoding='utf-8')
 service=next(x for x in fp.service if x.name=='HumanLearning');specs=[]
 for m in service.method:
  req=m.input_type.rsplit('.',1)[-1];resp=m.output_type.rsplit('.',1)[-1]
  specs.append(f'    {m.name!r}: (hitl__pb2.{req}, hitl__pb2.{resp}),')
 code='''# Generated-compatible gRPC bindings for baloot.v18.HumanLearning.\nimport grpc\nfrom . import hitl_pb2 as hitl__pb2\n_SPECS = {\n'''+"\n".join(specs)+'''\n}\nclass HumanLearningStub:\n    def __init__(self,channel):\n        for name,(request_type,response_type) in _SPECS.items():\n            setattr(self,name,channel.unary_unary(f"/baloot.v18.HumanLearning/{name}",request_serializer=request_type.SerializeToString,response_deserializer=response_type.FromString,_registered_method=True))\nclass HumanLearningServicer:\n    pass\ndef add_HumanLearningServicer_to_server(servicer,server):\n    handlers={}\n    for name,(request_type,response_type) in _SPECS.items():\n        handlers[name]=grpc.unary_unary_rpc_method_handler(getattr(servicer,name),request_deserializer=request_type.FromString,response_serializer=response_type.SerializeToString)\n    generic=grpc.method_handlers_generic_handler("baloot.v18.HumanLearning",handlers);server.add_generic_rpc_handlers((generic,))\n    if hasattr(server,"add_registered_method_handlers"): server.add_registered_method_handlers("baloot.v18.HumanLearning",handlers)\n'''
 (OUT/'hitl_pb2_grpc.py').write_text(code,encoding='utf-8');print(OUT)
if __name__=='__main__':main()

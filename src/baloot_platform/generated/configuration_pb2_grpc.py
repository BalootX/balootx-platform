# Generated-compatible gRPC bindings for baloot.v19.ConfigurationControl.
import grpc
from . import configuration_pb2 as configuration__pb2

_SPECS = {
    'GetCapabilities': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'ListProfiles': (configuration__pb2.ProfileQuery, configuration__pb2.JsonResult),
    'CreateProfile': (configuration__pb2.ProfileDraft, configuration__pb2.JsonResult),
    'PublishProfile': (configuration__pb2.Identifier, configuration__pb2.JsonResult),
    'RejectProfile': (configuration__pb2.TransitionRequest, configuration__pb2.JsonResult),
    'RetireProfile': (configuration__pb2.TransitionRequest, configuration__pb2.JsonResult),
    'ResolveProfile': (configuration__pb2.ProfileResolution, configuration__pb2.JsonResult),
    'CompareProfiles': (configuration__pb2.ProfileCompare, configuration__pb2.JsonResult),
    'ListBundles': (configuration__pb2.BundleQuery, configuration__pb2.JsonResult),
    'CreateBundle': (configuration__pb2.BundleDraft, configuration__pb2.JsonResult),
    'PublishBundle': (configuration__pb2.Identifier, configuration__pb2.JsonResult),
    'RejectBundle': (configuration__pb2.TransitionRequest, configuration__pb2.JsonResult),
    'RetireBundle': (configuration__pb2.TransitionRequest, configuration__pb2.JsonResult),
    'ResolveBundle': (configuration__pb2.BundleResolution, configuration__pb2.JsonResult),
    'ResolveEffectiveRuntimeConfigV2': (configuration__pb2.BundleResolution, configuration__pb2.JsonResult),
    'RollbackBundle': (configuration__pb2.BundleRollback, configuration__pb2.JsonResult),
    'ListPlugins': (configuration__pb2.PluginQuery, configuration__pb2.JsonResult),
    'GetPerformanceProfile': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'RunPerformanceQuick': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'GetCapabilityMatrix': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'GetCertificationProfile': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'VerifyCertificationCandidate': (configuration__pb2.JsonPayload, configuration__pb2.JsonResult),
    'GetBenchmarkProfile': (configuration__pb2.Empty, configuration__pb2.JsonResult),
    'EvaluateIntelligenceBenchmark': (configuration__pb2.JsonPayload, configuration__pb2.JsonResult),
    'GetOperationsProfile': (configuration__pb2.Empty, configuration__pb2.JsonResult),
}

class ConfigurationControlStub:
    def __init__(self, channel):
        for name,(request_type,response_type) in _SPECS.items():
            setattr(self,name,channel.unary_unary(f'/baloot.v19.ConfigurationControl/{name}',request_serializer=request_type.SerializeToString,response_deserializer=response_type.FromString,_registered_method=True))

class ConfigurationControlServicer:
    pass

def add_ConfigurationControlServicer_to_server(servicer, server):
    handlers={}
    for name,(request_type,response_type) in _SPECS.items():
        handlers[name]=grpc.unary_unary_rpc_method_handler(getattr(servicer,name),request_deserializer=request_type.FromString,response_serializer=response_type.SerializeToString)
    generic=grpc.method_handlers_generic_handler('baloot.v19.ConfigurationControl',handlers)
    server.add_generic_rpc_handlers((generic,))
    if hasattr(server,'add_registered_method_handlers'):
        server.add_registered_method_handlers('baloot.v19.ConfigurationControl',handlers)

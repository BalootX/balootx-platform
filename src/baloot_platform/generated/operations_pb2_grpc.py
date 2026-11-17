# Generated-compatible gRPC bindings for baloot.v17.BalootOperations.
import grpc
from . import operations_pb2 as operations__pb2

_SPECS = {
    'HealthCheck': (operations__pb2.Empty, operations__pb2.Health),
    'CreateLiveRoom': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'GetLiveRoom': (operations__pb2.Identifier, operations__pb2.JsonResponse),
    'ClaimLiveSeat': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'SetLiveReady': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'StartLiveTable': (operations__pb2.Identifier, operations__pb2.JsonResponse),
    'SubmitLiveAction': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'VerifyLiveRoom': (operations__pb2.Identifier, operations__pb2.JsonResponse),
    'OpenAdjudicationCase': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'GetAdjudicationCase': (operations__pb2.Identifier, operations__pb2.JsonResponse),
    'ListAdjudicationCases': (operations__pb2.JsonRequest, operations__pb2.JsonResponse),
    'RecommendAdjudicationRuling': (operations__pb2.Identifier, operations__pb2.JsonResponse),
    'SecurityCapabilities': (operations__pb2.Empty, operations__pb2.JsonResponse),
}

class BalootOperationsStub:
    def __init__(self, channel):
        for name,(request_type,response_type) in _SPECS.items():
            setattr(self,name,channel.unary_unary(
                f"/baloot.v17.BalootOperations/{name}",
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
                _registered_method=True,
            ))

class BalootOperationsServicer:
    pass

def add_BalootOperationsServicer_to_server(servicer,server):
    handlers={}
    for name,(request_type,response_type) in _SPECS.items():
        handlers[name]=grpc.unary_unary_rpc_method_handler(
            getattr(servicer,name),
            request_deserializer=request_type.FromString,
            response_serializer=response_type.SerializeToString,
        )
    generic=grpc.method_handlers_generic_handler("baloot.v17.BalootOperations",handlers)
    server.add_generic_rpc_handlers((generic,))
    if hasattr(server,"add_registered_method_handlers"):
        server.add_registered_method_handlers("baloot.v17.BalootOperations",handlers)

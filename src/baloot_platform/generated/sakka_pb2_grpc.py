# Generated-compatible gRPC bindings for baloot.v8.SakkaPlatform.
# Do not edit by hand.
import grpc
from . import sakka_pb2 as sakka__pb2

_SPECS = {
    'ListBoards': (sakka__pb2.ListBoardsRequest, sakka__pb2.JsonResult),
    'CreateBoard': (sakka__pb2.JsonPayload, sakka__pb2.JsonResult),
    'GetBoard': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'VerifyBoard': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'CreateRoom': (sakka__pb2.JsonPayload, sakka__pb2.JsonResult),
    'GetRoom': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'JoinRoom': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'StartRun': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'GetRun': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'VerifyRun': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
    'PlayAction': (sakka__pb2.RunActionRequest, sakka__pb2.JsonResult),
    'AuditRun': (sakka__pb2.RunAuditRequest, sakka__pb2.JsonResult),
    'ShareDecision': (sakka__pb2.ShareDecisionRequest, sakka__pb2.JsonResult),
    'RecordFeedback': (sakka__pb2.FeedbackRequest, sakka__pb2.JsonResult),
    'GetLeaderboard': (sakka__pb2.Identifier, sakka__pb2.JsonResult),
}

class SakkaPlatformStub:
    def __init__(self, channel):
        for name, (request_type, response_type) in _SPECS.items():
            setattr(self, name, channel.unary_unary(
                f'/baloot.v8.SakkaPlatform/{name}',
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
                _registered_method=True,
            ))

class SakkaPlatformServicer:
    pass

def add_SakkaPlatformServicer_to_server(servicer, server):
    handlers = {}
    for name, (request_type, response_type) in _SPECS.items():
        method = getattr(servicer, name)
        handlers[name] = grpc.unary_unary_rpc_method_handler(
            method, request_deserializer=request_type.FromString,
            response_serializer=response_type.SerializeToString,
        )
    generic = grpc.method_handlers_generic_handler('baloot.v8.SakkaPlatform', handlers)
    server.add_generic_rpc_handlers((generic,))
    if hasattr(server, 'add_registered_method_handlers'):
        server.add_registered_method_handlers('baloot.v8.SakkaPlatform', handlers)

# Generated-compatible gRPC bindings for baloot.v9.SharedPlatform.
# Do not edit by hand.
import grpc
from . import shared_pb2 as shared__pb2

_SPECS = {
    'ListBoards': (shared__pb2.ListBoardsRequest, shared__pb2.JsonResult),
    'CreateBoard': (shared__pb2.JsonPayload, shared__pb2.JsonResult),
    'GetBoard': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'VerifyBoard': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'CreateRoom': (shared__pb2.JsonPayload, shared__pb2.JsonResult),
    'GetRoom': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'JoinRoom': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'StartRun': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'GetRun': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'VerifyRun': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'PlayAction': (shared__pb2.RunActionRequest, shared__pb2.JsonResult),
    'GetRunReview': (shared__pb2.RunReviewRequest, shared__pb2.JsonResult),
    'ExplainDecision': (shared__pb2.ExplainDecisionRequest, shared__pb2.JsonResult),
    'GetDecisionReviews': (shared__pb2.DecisionRequest, shared__pb2.JsonResult),
    'SubmitDecisionReview': (shared__pb2.ReviewDecisionRequest, shared__pb2.JsonResult),
    'ShareDecision': (shared__pb2.ShareDecisionRequest, shared__pb2.JsonResult),
    'GetLeaderboard': (shared__pb2.Identifier, shared__pb2.JsonResult),
    'CompareRuns': (shared__pb2.CompareRunsRequest, shared__pb2.JsonResult),
}

class SharedPlatformStub:
    def __init__(self, channel):
        for name, (request_type, response_type) in _SPECS.items():
            setattr(self, name, channel.unary_unary(
                f'/baloot.v9.SharedPlatform/{name}',
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
                _registered_method=True,
            ))

class SharedPlatformServicer:
    pass

def add_SharedPlatformServicer_to_server(servicer, server):
    handlers = {}
    for name, (request_type, response_type) in _SPECS.items():
        method = getattr(servicer, name)
        handlers[name] = grpc.unary_unary_rpc_method_handler(
            method, request_deserializer=request_type.FromString,
            response_serializer=response_type.SerializeToString,
        )
    generic = grpc.method_handlers_generic_handler('baloot.v9.SharedPlatform', handlers)
    server.add_generic_rpc_handlers((generic,))
    if hasattr(server, 'add_registered_method_handlers'):
        server.add_registered_method_handlers('baloot.v9.SharedPlatform', handlers)

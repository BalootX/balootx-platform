# Generated-compatible gRPC bindings for baloot.v18.HumanLearning.
import grpc
from . import hitl_pb2 as hitl__pb2
_SPECS = {
    'HealthCheck': (hitl__pb2.Empty, hitl__pb2.Health),
    'Profile': (hitl__pb2.Empty, hitl__pb2.JsonResponse),
    'ListCases': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'CreateCase': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'GetCase': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'AssignBlindReview': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'SubmitReview': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'ComputeConsensus': (hitl__pb2.Identifier, hitl__pb2.JsonResponse),
    'SetConsent': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'GetConsent': (hitl__pb2.Identifier, hitl__pb2.JsonResponse),
    'CreateTrainingCandidate': (hitl__pb2.Identifier, hitl__pb2.JsonResponse),
    'RecordChallengerEvaluation': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'AcceptTrainingCandidate': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'CreateImprovementProposal': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'EvaluateImprovement': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'ApproveImprovement': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'FinalizeImprovement': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
    'RenderReviewClip': (hitl__pb2.JsonRequest, hitl__pb2.JsonResponse),
}
class HumanLearningStub:
    def __init__(self,channel):
        for name,(request_type,response_type) in _SPECS.items():
            setattr(self,name,channel.unary_unary(f"/baloot.v18.HumanLearning/{name}",request_serializer=request_type.SerializeToString,response_deserializer=response_type.FromString,_registered_method=True))
class HumanLearningServicer:
    pass
def add_HumanLearningServicer_to_server(servicer,server):
    handlers={}
    for name,(request_type,response_type) in _SPECS.items():
        handlers[name]=grpc.unary_unary_rpc_method_handler(getattr(servicer,name),request_deserializer=request_type.FromString,response_serializer=response_type.SerializeToString)
    generic=grpc.method_handlers_generic_handler("baloot.v18.HumanLearning",handlers);server.add_generic_rpc_handlers((generic,))
    if hasattr(server,"add_registered_method_handlers"): server.add_registered_method_handlers("baloot.v18.HumanLearning",handlers)

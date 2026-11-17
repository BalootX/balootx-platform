# Generated-compatible gRPC bindings for baloot.v2.BalootPlatform.
# Do not edit by hand.
import grpc
from . import platform_pb2 as platform__pb2


_SPECS = {
    'HealthCheck': (platform__pb2.Empty, platform__pb2.Health),
    'CreateGame': (platform__pb2.CreateGameRequest, platform__pb2.GameState),
    'GetGame': (platform__pb2.GameRef, platform__pb2.GameState),
    'SubmitBid': (platform__pb2.BidAction, platform__pb2.GameState),
    'PlayCard': (platform__pb2.PlayAction, platform__pb2.GameState),
    'ValidateContext': (platform__pb2.ContextRequest, platform__pb2.GameState),
    'ContinueContext': (platform__pb2.ContextRequest, platform__pb2.GameState),
    'Analyze': (platform__pb2.AnalysisRequest, platform__pb2.AnalysisResult),
    'CreateVideo': (platform__pb2.VideoJobRequest, platform__pb2.JobRef),
    'GetJob': (platform__pb2.JobRef, platform__pb2.JobResult),
    'GetReplay': (platform__pb2.Identifier, platform__pb2.ReplayPackage),
    'ReopenGame': (platform__pb2.ReopenGameRequest, platform__pb2.HistoryOperationResult),
    'ForkDecision': (platform__pb2.ForkDecisionRequest, platform__pb2.HistoryOperationResult),
    'AuditGame': (platform__pb2.ReplayAuditRequest, platform__pb2.AuditResult),
    'ExplainDecision': (platform__pb2.DecisionAuditRequest, platform__pb2.AuditResult),
    'CompareDecision': (platform__pb2.DecisionRef, platform__pb2.AuditResult),
    'GetDecisionDifficulty': (platform__pb2.DecisionRef, platform__pb2.DifficultyResult),
    'RecordDecisionAttempt': (platform__pb2.DifficultyAttempt, platform__pb2.DifficultyResult),
    'GetChallengeDifficulty': (platform__pb2.ChallengeRef, platform__pb2.DifficultyResult),
    'RecordChallengeAttempt': (platform__pb2.ChallengeAttempt, platform__pb2.DifficultyResult),
    'GetAdaptiveChallenge': (platform__pb2.AdaptiveChallengeRequest, platform__pb2.AnalysisResult),
    'GetVideoRecord': (platform__pb2.VideoRecordRef, platform__pb2.VideoRecord),
    'GetVideoTimeline': (platform__pb2.VideoRecordRef, platform__pb2.VideoRecord),
    'GetVideoAuditMap': (platform__pb2.VideoRecordRef, platform__pb2.VideoRecord),
    'RenderGameReplay': (platform__pb2.HistoricalReplayVideoRequest, platform__pb2.JobRef),
    'CreateRenderBatch': (platform__pb2.RenderBatchRequest, platform__pb2.JobRef),
    'GetRenderBatch': (platform__pb2.JobRef, platform__pb2.JobResult),
    'CancelRenderBatch': (platform__pb2.JobRef, platform__pb2.JobResult),
}


class BalootPlatformStub:
    def __init__(self, channel):
        for name, (request_type, response_type) in _SPECS.items():
            setattr(self, name, channel.unary_unary(
                f"/baloot.v2.BalootPlatform/{name}",
                request_serializer=request_type.SerializeToString,
                response_deserializer=response_type.FromString,
                _registered_method=True,
            ))


class BalootPlatformServicer:
    pass


def add_BalootPlatformServicer_to_server(servicer, server):
    handlers = {}
    for name, (request_type, response_type) in _SPECS.items():
        method = getattr(servicer, name)
        handlers[name] = grpc.unary_unary_rpc_method_handler(
            method,
            request_deserializer=request_type.FromString,
            response_serializer=response_type.SerializeToString,
        )
    generic = grpc.method_handlers_generic_handler("baloot.v2.BalootPlatform", handlers)
    server.add_generic_rpc_handlers((generic,))
    if hasattr(server, "add_registered_method_handlers"):
        server.add_registered_method_handlers("baloot.v2.BalootPlatform", handlers)

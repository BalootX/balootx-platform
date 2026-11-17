from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    id: str
    kind: str
    version: str
    description: str
    capabilities: tuple[str, ...] = ()
    resource_class: str = "cpu"
    deterministic: bool = True
    api_version: str = "1"
    host_api_range: str = ">=1,<2"
    profile_schema_range: str = ">=1,<2"
    isolation: str = "in_process"
    permissions: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    artifact_sha256: str | None = None
    signed: bool = False
    timeout_seconds: int = 30
    maximum_memory_mb: int = 512
    trust_class: str = "builtin"
    compatible_profile_schemas: tuple[str, ...] = ()
    signature: str | None = None
    contract_id: str | None = None
    contract_version: str | None = None
    safety_guards: tuple[str, ...] = ()
    cancellation_mode: str = "bounded"
    failure_mode: str = "fail_closed"
    rollback_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = {
            "id": self.id, "kind": self.kind, "version": self.version, "apiVersion": self.api_version,
            "description": self.description, "capabilities": list(self.capabilities),
            "resourceClass": self.resource_class, "deterministic": self.deterministic,
            "hostApiRange": self.host_api_range, "profileSchemaRange": self.profile_schema_range,
            "isolation": self.isolation, "permissions": list(self.permissions),
            "sideEffects": list(self.side_effects), "artifactSha256": self.artifact_sha256,
            "signed": self.signed, "timeoutSeconds": self.timeout_seconds,
            "maximumMemoryMb": self.maximum_memory_mb, "trustClass": self.trust_class,
            "compatibleProfileSchemas": list(self.compatible_profile_schemas), "signature": self.signature,
        }
        if self.contract_id is not None:
            value["contract"] = {
                "id": self.contract_id,
                "version": self.contract_version,
                "safetyGuards": list(self.safety_guards),
                "cancellationMode": self.cancellation_mode,
                "failureMode": self.failure_mode,
                "rollbackSupported": self.rollback_supported,
            }
        return value


@dataclass(frozen=True, slots=True)
class RenderArtifacts:
    video: Path
    poster: Path
    episode_json: Path
    captions: Path
    contact_sheet: Path
    manifest: Path
    timeline: Path | None = None
    semantic_files: tuple[Path, ...] = ()
    captions_vtt: Path | None = None
    transcript: Path | None = None


@runtime_checkable
class Plugin(Protocol):
    descriptor: PluginDescriptor


@runtime_checkable
class VideoRendererPlugin(Plugin, Protocol):
    def execute(self, request: Any, **kwargs: Any) -> RenderArtifacts: ...


@runtime_checkable
class VideoCompositorPlugin(Plugin, Protocol):
    def execute(self, request: Any, **kwargs: Any) -> RenderArtifacts: ...


@runtime_checkable
class BiddingStrategyPlugin(Plugin, Protocol):
    def choose_bid(self, state: Any, legal_actions: tuple[str, ...], **kwargs: Any) -> tuple[str, Any | None]: ...


@runtime_checkable
class ResearchAuctionSelectorPlugin(Plugin, Protocol):
    """Optional actor-safe Research override for one auction decision."""

    def choose_with_trace(
        self,
        state: Any,
        machine: Any,
        rules: Any,
        **kwargs: Any,
    ) -> tuple[str, Any | None, dict[str, Any]]: ...


@runtime_checkable
class PlayStrategyPlugin(Plugin, Protocol):
    def choose_card(self, engine: Any, state: Any, legal_cards: tuple[Any, ...], **kwargs: Any) -> Any: ...


@runtime_checkable
class ResearchPlaySelectorPlugin(Plugin, Protocol):
    """Optional actor-safe Research override for one non-forced play decision."""

    def choose(
        self,
        *,
        engine: Any,
        state: Any,
        legal_cards: tuple[Any, ...],
        seed: int,
        cancel_check: Any | None = None,
    ) -> tuple[Any | None, dict[str, Any]] | None: ...

    def provenance(self) -> dict[str, Any]: ...


@runtime_checkable
class AnalysisPlugin(Plugin, Protocol):
    def analyze(self, information_state: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TraceAnalyzerPlugin(Plugin, Protocol):
    def analyze_trace(self, trace: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class AnalysisFunnelPlugin(Plugin, Protocol):
    def process(self, trace_references: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ArtifactStorePlugin(Plugin, Protocol):
    def put(self, data: bytes, *, media_type: str, name: str | None = None) -> dict[str, Any]: ...
    def put_file(self, path: str | Path, *, media_type: str, name: str | None = None) -> dict[str, Any]: ...
    def get(self, reference: str) -> bytes: ...
    def materialize(self, reference: str, *, suffix: str = "", cache_dir: str | Path | None = None) -> Path: ...


@runtime_checkable
class HistoryStorePlugin(Plugin, Protocol):
    """Immutable game, decision, analysis, and lineage persistence boundary."""

    def archive_game(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def get_game(self, identifier: str) -> dict[str, Any] | None: ...
    def verify_game(self, identifier: str) -> dict[str, Any]: ...
    def list_games(self, *, query: str = "", offset: int = 0, limit: int = 100) -> list[dict[str, Any]]: ...


@runtime_checkable
class LearningBackendPlugin(Plugin, Protocol):
    """Human-feedback, dataset, model-training, and promotion boundary."""

    def reanalyze_decision(self, decision_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def add_feedback(self, decision_id: str, **kwargs: Any) -> dict[str, Any]: ...
    def build_dataset(self, **kwargs: Any) -> dict[str, Any]: ...
    def train_model(self, dataset_id: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class ChallengeMinerPlugin(Plugin, Protocol):
    """Progressive, resource-bounded challenge discovery boundary."""

    def mine(self, request: Any) -> list[dict[str, Any]]: ...


@runtime_checkable
class GeneratorPlugin(Plugin, Protocol):
    """Deterministic game/deal generation boundary.

    Generators may choose seeds or legal search lanes, but they never bypass
    the authoritative engine, legality checks, scoring, or observer safety.
    """

    def generate(self, seed: int, **kwargs: Any) -> Any: ...


@runtime_checkable
class DealCandidateGeneratorPlugin(Plugin, Protocol):
    """Construct complete deck candidates without owning Baloot truth."""

    def generate(self, spec: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TacticalDetectorPlugin(Plugin, Protocol):
    def detect(self, selected: Any, alternatives: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TacticalVerifierPlugin(Plugin, Protocol):
    def verify_decision(self, snapshot: Any, actor_analysis: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PerspectiveAnalyzerPlugin(Plugin, Protocol):
    def analyze_decision(self, decision: Any, audit: Any, **kwargs: Any) -> Any: ...
    def analyze_laffa(self, identifier: str, audit_report: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ChallengeScorerPlugin(Plugin, Protocol):
    def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class SelectionPolicyPlugin(Plugin, Protocol):
    def select(self, candidates: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ContextSearchPlugin(Plugin, Protocol):
    def search(self, laffa: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class FinalistReplayPlugin(Plugin, Protocol):
    def replay(self, parent: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class SakkaComposerPlugin(Plugin, Protocol):
    def compose(self, laffat: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ExplanationPlugin(Plugin, Protocol):
    def explain(self, evidence: Any, locale: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class AudioPlugin(Plugin, Protocol):
    def render_audio(self, script: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PublisherPlugin(Plugin, Protocol):
    def publish(self, publication: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class BatchExecutorPlugin(Plugin, Protocol):
    def execute_batch(self, shard: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ModelInferencePlugin(Plugin, Protocol):
    """Observer-safe policy/value/belief inference boundary."""

    def predict(self, features: Any, legal_mask: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class GameRepositoryPlugin(Plugin, Protocol):
    """Versioned authoritative game-session persistence boundary."""

    def save(self, session: Any) -> Any: ...
    def load(self, game_id: str) -> Any | None: ...
    def load_version(self, game_id: str, version: int) -> Any | None: ...
    def verify(self, game_id: str) -> dict[str, Any]: ...


@runtime_checkable
class ReplayAuditPlugin(Plugin, Protocol):
    """Observer-safe, versioned decision and game audit boundary."""

    def explain_decision(self, decision_id: str, **kwargs: Any) -> Any: ...
    def audit_game(self, identifier: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class DifficultyCalibratorPlugin(Plugin, Protocol):
    """Audience-response calibration and adaptive-delivery boundary."""

    def record_attempt(self, **kwargs: Any) -> dict[str, Any]: ...
    def recalibrate(self, decision_id: str, **kwargs: Any) -> dict[str, Any]: ...


@runtime_checkable
class DomainPrimitivePlugin(Plugin, Protocol):
    """Provide one governed, versioned Baloot primitive definition."""

    def definition(self) -> Any: ...


@runtime_checkable
class ReferenceLoweringPlugin(Plugin, Protocol):
    """Lower a governed primitive into the clear reference interpreter."""

    def lower_reference(self, primitive: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class RuntimeLoweringPlugin(Plugin, Protocol):
    """Lower a governed primitive into the optimized runtime kernel."""

    def lower_runtime(self, primitive: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class CpSatLoweringPlugin(Plugin, Protocol):
    """Compile governed domain input into a bounded CP-SAT result."""

    def compile(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class InformationSetPlugin(Plugin, Protocol):
    """Build an actor-safe information partition from authoritative state."""

    def partition(self, state: Any, actor: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class BeliefProfilePlugin(Plugin, Protocol):
    """Build a deterministic observer-safe belief policy from frozen input."""

    def build(self, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ResearchObjectivePlugin(Plugin, Protocol):
    """Score immutable evidence without overriding a hard qualification gate."""

    def score(self, evidence: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class BestResponsePlugin(Plugin, Protocol):
    """Prove a bounded actor-safe best response over authoritative actions."""

    def prove(self, problem: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class TacticalProofPlugin(Plugin, Protocol):
    """Derive causal tactical evidence without changing semantic truth."""

    def detect(self, selected: Any, alternatives: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class DiversityPolicyPlugin(Plugin, Protocol):
    """Select qualified evidence without weakening upstream gates."""

    def select(self, candidates: Any, profile: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class ResearchExecutorPlugin(Plugin, Protocol):
    """Launch and cancel a frozen, idempotent Research execution."""

    def submit(self, recipe_code: str, *, idempotency_key: str, **kwargs: Any) -> Any: ...
    def cancel(self, run_code: str) -> Any: ...


@runtime_checkable
class ResearchRenderAdapterPlugin(Plugin, Protocol):
    """Prepare replay-valid Research media only after an explicit request."""

    def prepare(self, result_code: str, **kwargs: Any) -> Any: ...

"""Content-addressed plans for isolated batch and multi-angle rendering.

Multi-angle plans intentionally reference separate ``RenderBundleV1`` values.
They never merge private observer layers into one transport document.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from baloot_platform.audit.models import content_hash

from .render_bundle import RenderBundleV1


_SEATS = frozenset({"north", "west", "south", "east"})
_RENDITIONS = frozenset(
    {"actor_challenge", "partner_inference", "opponent_prediction", "seat_narrative", "post_reveal_analysis"}
)
_TRANSITIONS = frozenset({"cut", "fade"})


@dataclass(frozen=True, slots=True)
class RenderBatchItemV1:
    ordinal: int
    bundle_sha256: str
    export_preset: str
    observer_seat: str
    basename: str
    proof_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("render batch ordinal must be non-negative")
        if len(self.bundle_sha256) != 64 or any(value not in "0123456789abcdef" for value in self.bundle_sha256):
            raise ValueError("render batch bundle identity must be a lowercase SHA-256")
        if self.export_preset not in {"hooked_flagship", "decision_clip"}:
            raise ValueError("render batch export preset is unsupported")
        if self.observer_seat not in _SEATS:
            raise ValueError("render batch observer seat is invalid")
        if not self.basename or any(value in self.basename for value in '\\/:*?"<>|'):
            raise ValueError("render batch basename is invalid")
        if self.proof_artifact_sha256 is not None and (
            len(self.proof_artifact_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.proof_artifact_sha256
            )
        ):
            raise ValueError(
                "render batch proof artifact identity must be a lowercase SHA-256"
            )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "ordinal": self.ordinal,
            "bundleSha256": self.bundle_sha256,
            "exportPreset": self.export_preset,
            "observerSeat": self.observer_seat,
            "basename": self.basename,
        }
        if self.proof_artifact_sha256 is not None:
            value["proofArtifactSha256"] = self.proof_artifact_sha256
        return value


@dataclass(frozen=True, slots=True)
class RenderBatchV1:
    items: tuple[RenderBatchItemV1, ...]
    sha256: str

    @classmethod
    def create(
        cls,
        requests: Sequence[tuple[RenderBundleV1, str]],
        *,
        proof_artifact_hashes: Mapping[str, str] | None = None,
    ) -> "RenderBatchV1":
        if not requests:
            raise ValueError("render batch requires at least one bundle")
        items: list[RenderBatchItemV1] = []
        seen_hashes: set[str] = set()
        seen_basenames: set[str] = set()
        artifact_hashes = dict(proof_artifact_hashes or {})
        for ordinal, (bundle, basename) in enumerate(requests):
            if not bundle.verify():
                raise ValueError("render batch contains an invalid bundle")
            if bundle.bundle_sha256 in seen_hashes or basename in seen_basenames:
                raise ValueError("render batch bundle hashes and basenames must be unique")
            seen_hashes.add(bundle.bundle_sha256)
            seen_basenames.add(basename)
            items.append(RenderBatchItemV1(
                ordinal=ordinal,
                bundle_sha256=bundle.bundle_sha256,
                export_preset=str(bundle.value["exportPreset"]),
                observer_seat=str(bundle.value["observerSeat"]),
                basename=str(basename),
                proof_artifact_sha256=artifact_hashes.pop(
                    bundle.bundle_sha256,
                    None,
                ),
            ))
        if artifact_hashes:
            raise ValueError("render batch proof artifacts reference unknown bundles")
        core = {"schemaVersion": 1, "items": [item.to_dict() for item in items]}
        return cls(tuple(items), content_hash(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "items": [item.to_dict() for item in self.items],
            "sha256": self.sha256,
        }

    def verify(self, bundles: Mapping[str, RenderBundleV1]) -> bool:
        core = {"schemaVersion": 1, "items": [item.to_dict() for item in self.items]}
        if content_hash(core) != self.sha256:
            return False
        if tuple(item.ordinal for item in self.items) != tuple(range(len(self.items))):
            return False
        if len({item.bundle_sha256 for item in self.items}) != len(self.items):
            return False
        if len({item.basename for item in self.items}) != len(self.items):
            return False
        for item in self.items:
            bundle = bundles.get(item.bundle_sha256)
            if bundle is None or not bundle.verify():
                return False
            if bundle.value["exportPreset"] != item.export_preset:
                return False
            if bundle.value["observerSeat"] != item.observer_seat:
                return False
        return True


@dataclass(frozen=True, slots=True)
class MultiAngleSegmentV1:
    ordinal: int
    bundle_sha256: str
    observer_seat: str
    rendition_type: str
    enrollment_uri: str
    source_start_frame: int
    source_end_frame: int
    transition: str
    transition_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "bundleSha256": self.bundle_sha256,
            "observerSeat": self.observer_seat,
            "renditionType": self.rendition_type,
            "enrollmentUri": self.enrollment_uri,
            "sourceStartFrame": self.source_start_frame,
            "sourceEndFrame": self.source_end_frame,
            "transition": self.transition,
            "transitionFrames": self.transition_frames,
        }


@dataclass(frozen=True, slots=True)
class MultiAngleRenderPlanV1:
    segments: tuple[MultiAngleSegmentV1, ...]
    fps: int
    width: int
    height: int
    total_frames: int
    sha256: str

    @classmethod
    def create(
        cls,
        requests: Sequence[tuple[RenderBundleV1, str, int, int, str, int]],
    ) -> "MultiAngleRenderPlanV1":
        """Build an explicit composition plan from separately safe bundles.

        Each request is ``(bundle, rendition, start, end, transition,
        transition_frames)``. Frame ranges are source-bundle ranges.
        """
        if len(requests) < 2:
            raise ValueError("multi-angle rendering requires at least two isolated segments")
        first_render = requests[0][0].value["render"]
        fps = int(first_render["fps"])
        width = int(first_render["width"])
        height = int(first_render["height"])
        segments: list[MultiAngleSegmentV1] = []
        total_frames = 0
        for ordinal, (bundle, rendition, start, end, transition, transition_frames) in enumerate(requests):
            if not bundle.verify():
                raise ValueError("multi-angle segment contains an invalid bundle")
            render = bundle.value["render"]
            if (int(render["fps"]), int(render["width"]), int(render["height"])) != (fps, width, height):
                raise ValueError("multi-angle segments must use the same media profile")
            if rendition not in _RENDITIONS:
                raise ValueError("multi-angle rendition type is unsupported")
            if transition not in _TRANSITIONS:
                raise ValueError("multi-angle transition is unsupported")
            if ordinal == 0 and (transition != "cut" or transition_frames != 0):
                raise ValueError("the first multi-angle segment cannot have an inbound transition")
            if ordinal > 0 and transition == "cut" and transition_frames != 0:
                raise ValueError("multi-angle cut transitions cannot consume frames")
            if ordinal > 0 and transition == "fade" and transition_frames <= 0:
                raise ValueError("multi-angle fade transitions require a positive frame count")
            maximum = int(bundle.value["timeline"]["totalFrames"])
            if start < 0 or end <= start or end > maximum:
                raise ValueError("multi-angle source frame window is invalid")
            duration = end - start
            if ordinal > 0 and transition_frames >= min(
                duration,
                segments[-1].source_end_frame - segments[-1].source_start_frame,
            ):
                raise ValueError("multi-angle transition must be shorter than adjacent segments")
            identity = bundle.value["identity"]
            segment = MultiAngleSegmentV1(
                ordinal=ordinal,
                bundle_sha256=bundle.bundle_sha256,
                observer_seat=str(bundle.value["observerSeat"]),
                rendition_type=rendition,
                enrollment_uri=str(identity["enrollmentUri"]),
                source_start_frame=int(start),
                source_end_frame=int(end),
                transition=transition,
                transition_frames=int(transition_frames),
            )
            segments.append(segment)
            total_frames += end - start
        if len({item.observer_seat for item in segments}) < 2:
            raise ValueError("multi-angle rendering requires at least two observer perspectives")
        core = {
            "schemaVersion": 1,
            "observerIsolation": "separate_render_bundles",
            "fps": fps,
            "canvas": {"width": width, "height": height, "aspectRatio": "9:16"},
            "segments": [item.to_dict() for item in segments],
            "totalFrames": total_frames,
        }
        return cls(tuple(segments), fps, width, height, total_frames, content_hash(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "observerIsolation": "separate_render_bundles",
            "fps": self.fps,
            "canvas": {"width": self.width, "height": self.height, "aspectRatio": "9:16"},
            "segments": [item.to_dict() for item in self.segments],
            "totalFrames": self.total_frames,
            "sha256": self.sha256,
        }

    def verify(self, bundles: Mapping[str, RenderBundleV1]) -> bool:
        try:
            rebuilt = self.create([
                (
                    bundles[item.bundle_sha256], item.rendition_type,
                    item.source_start_frame, item.source_end_frame,
                    item.transition, item.transition_frames,
                )
                for item in self.segments
            ])
        except (KeyError, TypeError, ValueError):
            return False
        return rebuilt.to_dict() == self.to_dict()

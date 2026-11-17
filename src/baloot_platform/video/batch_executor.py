"""Governed, idempotent execution of content-addressed render batches."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event
from typing import Any, Callable, Mapping

from baloot_platform.audit.models import content_hash
from baloot_platform.plugins.base import PluginDescriptor, VideoRendererPlugin

from .render_bundle import RenderBundleV1
from .render_plan import RenderBatchItemV1, RenderBatchV1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RenderBatchItemResultV1:
    ordinal: int
    bundle_sha256: str
    status: str
    manifest_name: str | None
    manifest_sha256: str | None
    error_code: str | None = None
    error_detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "bundleSha256": self.bundle_sha256,
            "status": self.status,
            "manifestName": self.manifest_name,
            "manifestSha256": self.manifest_sha256,
            "errorCode": self.error_code,
            "errorDetail": self.error_detail,
        }


@dataclass(frozen=True, slots=True)
class RenderBatchExecutionV1:
    batch_sha256: str
    status: str
    items: tuple[RenderBatchItemResultV1, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        batch_sha256: str,
        items: tuple[RenderBatchItemResultV1, ...],
    ) -> "RenderBatchExecutionV1":
        if not items:
            raise ValueError("render batch execution requires item evidence")
        statuses = {item.status for item in items}
        if "failed" in statuses:
            status = "failed"
        elif "cancelled" in statuses:
            status = "cancelled"
        elif statuses == {"completed"}:
            status = "completed"
        else:
            raise ValueError("render batch execution contains an unsupported status")
        core = {
            "schemaVersion": 1,
            "batchSha256": batch_sha256,
            "status": status,
            "items": [item.to_dict() for item in items],
        }
        return cls(batch_sha256, status, items, content_hash(core))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "batchSha256": self.batch_sha256,
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "sha256": self.sha256,
        }

    def verify(self) -> bool:
        try:
            rebuilt = self.build(self.batch_sha256, self.items)
        except ValueError:
            return False
        return rebuilt.to_dict() == self.to_dict()


class RenderBatchExecutorV1:
    descriptor = PluginDescriptor(
        id="render_batch_executor_v1",
        kind="batch_executor",
        version="1.0.0",
        description="Bounded, cancellable and idempotent execution of observer-safe render bundles",
        capabilities=(
            "render_batch_v1", "bounded_concurrency", "cooperative_cancellation",
            "idempotent_outputs", "stable_result_order",
        ),
        resource_class="video-cpu-premium",
        deterministic=True,
        isolation="in_process",
        side_effects=("filesystem_output",),
        timeout_seconds=900,
        maximum_memory_mb=4096,
    )

    def __init__(
        self,
        renderer: VideoRendererPlugin,
        *,
        maximum_concurrency: int,
        maximum_render_slots: int | None = None,
    ):
        if not 1 <= int(maximum_concurrency) <= 16:
            raise ValueError("render batch concurrency must be between one and sixteen")
        render_slots = (
            int(maximum_concurrency)
            if maximum_render_slots is None
            else int(maximum_render_slots)
        )
        if not 1 <= render_slots <= 64:
            raise ValueError("render batch slots must be between one and sixty-four")
        self.renderer = renderer
        self.maximum_concurrency = int(maximum_concurrency)
        self.maximum_render_slots = render_slots
        self._available_render_slots = render_slots
        self._render_slots_changed = Condition()

    def _render_slots(self, bundle: RenderBundleV1) -> int:
        value = bundle.value.get("render", {}).get("concurrency")
        if type(value) is not int or not 1 <= value <= self.maximum_render_slots:
            raise ValueError(
                "render bundle concurrency exceeds the governed render-slot budget"
            )
        return value

    def _acquire_render_slots(
        self,
        slots: int,
        cancel_event: Event | None,
    ) -> bool:
        with self._render_slots_changed:
            while self._available_render_slots < slots:
                if cancel_event is not None and cancel_event.is_set():
                    return False
                self._render_slots_changed.wait(timeout=0.1)
            if cancel_event is not None and cancel_event.is_set():
                return False
            self._available_render_slots -= slots
            return True

    def _release_render_slots(self, slots: int) -> None:
        with self._render_slots_changed:
            self._available_render_slots += slots
            self._render_slots_changed.notify_all()

    @staticmethod
    def _existing_manifest(
        directory: Path,
        item: RenderBatchItemV1,
        *,
        playback_evidence: bool = False,
        outcome_required: bool = False,
    ) -> tuple[str, str] | None:
        path = directory / f"{item.basename}-manifest.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if value.get("bundleSha256") != item.bundle_sha256:
            return None
        if playback_evidence:
            checkpoints = value.get("playbackEvidence") or ()
            if item.export_preset == "decision_clip":
                phases = ["context", "question", "reveal", "continuation", "continuation_end"]
                if outcome_required:
                    phases.append("outcome")
                if (len(checkpoints) != len(phases) or any(not isinstance(row, Mapping) for row in checkpoints)
                        or [row.get("checkpointPhase") for row in checkpoints] != phases
                        or any(row.get("evidenceScope") != "decision_clip_state_and_encoded_frame_parity"
                               for row in checkpoints)):
                    return None
            else:
                if (len(checkpoints) not in (33, 34)
                        or any(not isinstance(row, Mapping) for row in checkpoints)
                        or [row.get("completedPlayCount") for row in checkpoints[:33]] != list(range(33))):
                    return None
                has_outcome = len(checkpoints) == 34
                if (outcome_required and not has_outcome) or (has_outcome and (
                    checkpoints[-1].get("checkpointPhase") != "outcome"
                    or checkpoints[-1].get("completedPlayCount") != 32
                )):
                    return None
        files = value.get("files")
        if not isinstance(files, Mapping) or not files:
            return None
        for evidence in files.values():
            if not isinstance(evidence, Mapping):
                return None
            artifact = directory / str(evidence.get("name") or "")
            if (
                not artifact.is_file()
                or artifact.stat().st_size != int(evidence.get("bytes", -1))
                or _sha256(artifact) != evidence.get("sha256")
            ):
                return None
        return path.name, _sha256(path)

    def _execute_item(
        self,
        item: RenderBatchItemV1,
        bundle: RenderBundleV1,
        directory: Path,
        cancel_event: Event | None,
        playback_evidence: bool,
    ) -> RenderBatchItemResultV1:
        if cancel_event is not None and cancel_event.is_set():
            return RenderBatchItemResultV1(
                item.ordinal, item.bundle_sha256, "cancelled", None, None,
            )
        outcome_required = any(segment["stage"] == "outcome" for segment in bundle.value["timeline"]["segments"])
        existing = self._existing_manifest(
            directory,
            item,
            playback_evidence=playback_evidence,
            # Historical full exports remain reusable; new renders verify outcome below.
            outcome_required=outcome_required and item.export_preset == "decision_clip",
        )
        if existing is not None:
            return RenderBatchItemResultV1(
                item.ordinal, item.bundle_sha256, "completed", existing[0], existing[1],
            )
        slots = self._render_slots(bundle)
        acquired = False
        try:
            if not self._acquire_render_slots(slots, cancel_event):
                return RenderBatchItemResultV1(
                    item.ordinal, item.bundle_sha256, "cancelled", None, None,
                )
            acquired = True
            artifacts = self.renderer.execute(
                bundle,
                export_preset=item.export_preset,
                output_dir=directory,
                basename=item.basename,
                cancel_event=cancel_event,
                playback_evidence=playback_evidence,
            )
            manifest = Path(artifacts.manifest)
            if manifest.parent.resolve() != directory.resolve():
                raise ValueError("renderer returned a manifest outside the governed batch directory")
            verified = self._existing_manifest(
                directory,
                item,
                playback_evidence=playback_evidence,
                outcome_required=outcome_required,
            )
            if verified is None:
                raise ValueError("renderer output manifest failed idempotency verification")
            return RenderBatchItemResultV1(
                item.ordinal, item.bundle_sha256, "completed", verified[0], verified[1],
            )
        except Exception as exc:  # The batch retains stable failure evidence.
            detail = str(exc).replace("\r\n", "\n")[-4000:] or type(exc).__name__
            return RenderBatchItemResultV1(
                item.ordinal,
                item.bundle_sha256,
                "cancelled" if isinstance(exc, RuntimeError) and "cancelled" in str(exc).lower() else "failed",
                None,
                None,
                type(exc).__name__,
                detail,
            )
        finally:
            if acquired:
                self._release_render_slots(slots)

    def execute_batch(
        self,
        batch: RenderBatchV1,
        *,
        bundles: Mapping[str, RenderBundleV1],
        output_root: str | Path,
        cancel_event: Event | None = None,
        item_completed: Callable[[RenderBatchItemResultV1, int, int], None] | None = None,
        playback_evidence: bool = False,
    ) -> RenderBatchExecutionV1:
        if type(playback_evidence) is not bool:
            raise TypeError("render batch playback evidence flag must be boolean")
        if not batch.verify(bundles):
            raise ValueError("render batch or bundle identities failed verification")
        for item in batch.items:
            self._render_slots(bundles[item.bundle_sha256])
        root = Path(output_root).resolve()
        directory = root / f"batch-{batch.sha256[:16]}"
        directory.mkdir(parents=True, exist_ok=True)
        results: list[RenderBatchItemResultV1] = []
        with ThreadPoolExecutor(
            max_workers=min(self.maximum_concurrency, len(batch.items)),
            thread_name_prefix="baloot-render-batch",
        ) as executor:
            futures = {
                executor.submit(
                    self._execute_item,
                    item,
                    bundles[item.bundle_sha256],
                    directory,
                    cancel_event,
                    playback_evidence,
                ): item.ordinal
                for item in batch.items
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if item_completed is not None:
                    item_completed(result, len(results), len(batch.items))
        evidence = RenderBatchExecutionV1.build(
            batch.sha256,
            tuple(sorted(results, key=lambda item: item.ordinal)),
        )
        path = directory / "render-batch-execution.json"
        temporary = directory / ".render-batch-execution.json.tmp"
        temporary.write_text(
            json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
        return evidence

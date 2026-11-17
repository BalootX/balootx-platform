from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import yaml

from baloot_platform.audit.models import content_hash
from baloot_platform.config import build_bundle
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.video.batch_executor import RenderBatchExecutorV1
from baloot_platform.video.media_validation import (
    REMOTION_RENDERER_VERSION,
    renderer_source_sha256,
)
from baloot_platform.video.render_bundle import RenderBundleV1
from baloot_platform.video.render_plan import RenderBatchV1
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundles(root: Path) -> tuple[RenderBundleV1, RenderBundleV1]:
    profile = yaml.safe_load((root / "configs/video/premium.yaml").read_text(encoding="utf-8"))
    renderer = RemotionProfessionalRenderer(profile, repository_root=root)
    episode = load_episode_spec(root / "examples/v7-outcome-flip-source.json", build_bundle(root).rules)
    bundles: list[RenderBundleV1] = []
    for preset in ("hooked_flagship", "decision_clip"):
        raw = renderer.build_render_request(episode, export_preset=preset).to_dict()
        raw["render"]["concurrency"] = 1
        raw["renderer"]["version"] = REMOTION_RENDERER_VERSION
        raw["renderer"]["sourceSha256"] = renderer_source_sha256(root)
        core = {key: value for key, value in raw.items() if key != "bundleSha256"}
        raw["bundleSha256"] = content_hash(core)
        bundle = RenderBundleV1(raw)
        assert bundle.verify()
        bundles.append(bundle)
    return bundles[0], bundles[1]


def _with_concurrency(bundle: RenderBundleV1, concurrency: int) -> RenderBundleV1:
    raw = bundle.to_dict()
    raw["render"]["concurrency"] = concurrency
    core = {key: value for key, value in raw.items() if key != "bundleSha256"}
    raw["bundleSha256"] = content_hash(core)
    rebuilt = RenderBundleV1(raw)
    assert rebuilt.verify()
    return rebuilt


class FakeRenderer:
    def __init__(self, *, fail_basename: str | None = None):
        self.calls: list[str] = []
        self.fail_basename = fail_basename
        self.playback_flags: list[bool] = []

    def execute(self, bundle, **kwargs):
        basename = str(kwargs["basename"])
        self.calls.append(basename)
        self.playback_flags.append(bool(kwargs.get("playback_evidence", False)))
        if basename == self.fail_basename:
            raise ValueError("governed fake failure")
        output = Path(kwargs["output_dir"])
        video = output / f"{basename}.mp4"
        video.write_bytes(bundle.bundle_sha256.encode("ascii"))
        manifest = output / f"{basename}-manifest.json"
        value = {
            "schemaVersion": 1,
            "bundleSha256": bundle.bundle_sha256,
            "files": {
                "video": {
                    "name": video.name,
                    "bytes": video.stat().st_size,
                    "sha256": _sha256(video),
                }
            },
        }
        if kwargs.get("playback_evidence"):
            value["playbackEvidence"] = ([
                {"checkpointPhase": phase, "evidenceScope": "decision_clip_state_and_encoded_frame_parity"}
                for phase in ("context", "question", "reveal", "continuation", "continuation_end") +
                    (("outcome",) if any(row["stage"] == "outcome" for row in bundle.value["timeline"]["segments"]) else ())
            ] if bundle.value["exportPreset"] == "decision_clip" else [
                {"completedPlayCount": count} for count in range(33)] +
                [{"completedPlayCount": 32, "checkpointPhase": "outcome"}])
        manifest.write_text(
            json.dumps(value, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return SimpleNamespace(manifest=manifest)


def test_render_batch_execution_is_ordered_idempotent_and_self_verifying(root: Path, tmp_path: Path) -> None:
    flagship, decision = _bundles(root)
    bundles = {item.bundle_sha256: item for item in (flagship, decision)}
    batch = RenderBatchV1.create(((flagship, "flagship"), (decision, "decision")))
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=2)

    first = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    assert first.verify() and first.status == "completed"
    assert [item.ordinal for item in first.items] == [0, 1]
    assert sorted(renderer.calls) == ["decision", "flagship"]
    evidence_path = tmp_path / f"batch-{batch.sha256[:16]}" / "render-batch-execution.json"
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == first.to_dict()

    renderer.calls.clear()
    second = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    assert second.to_dict() == first.to_dict()
    assert renderer.calls == []


def test_render_batch_executes_items_concurrently(root: Path, tmp_path: Path) -> None:
    flagship, decision = _bundles(root)
    barrier = Barrier(2)

    class BarrierRenderer(FakeRenderer):
        def execute(self, bundle, **kwargs):
            barrier.wait(timeout=2)
            return super().execute(bundle, **kwargs)

    bundles = {item.bundle_sha256: item for item in (flagship, decision)}
    batch = RenderBatchV1.create(((flagship, "flagship"), (decision, "decision")))
    result = RenderBatchExecutorV1(
        BarrierRenderer(), maximum_concurrency=2,
    ).execute_batch(batch, bundles=bundles, output_root=tmp_path)

    assert result.status == "completed"


def test_render_slot_budget_is_shared_across_simultaneous_batches(
    root: Path,
    tmp_path: Path,
) -> None:
    flagship, decision = (
        _with_concurrency(bundle, 2) for bundle in _bundles(root)
    )

    class BlockingRenderer(FakeRenderer):
        def __init__(self):
            super().__init__()
            self.started = Event()
            self.release = Event()
            self.lock = Lock()
            self.active = 0
            self.maximum_active = 0

        def execute(self, bundle, **kwargs):
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            self.started.set()
            self.release.wait(timeout=3)
            try:
                return super().execute(bundle, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

    renderer = BlockingRenderer()
    executor = RenderBatchExecutorV1(
        renderer,
        maximum_concurrency=2,
        maximum_render_slots=2,
    )
    batches = (
        RenderBatchV1.create(((flagship, "flagship"),)),
        RenderBatchV1.create(((decision, "decision"),)),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            executor.execute_batch,
            batches[0],
            bundles={flagship.bundle_sha256: flagship},
            output_root=tmp_path / "first",
        )
        assert renderer.started.wait(timeout=2)
        second = pool.submit(
            executor.execute_batch,
            batches[1],
            bundles={decision.bundle_sha256: decision},
            output_root=tmp_path / "second",
        )
        assert not second.done()
        renderer.release.set()
        assert first.result(timeout=3).status == "completed"
        assert second.result(timeout=3).status == "completed"
    assert renderer.maximum_active == 1


def test_render_slot_wait_honors_cancellation(root: Path, tmp_path: Path) -> None:
    flagship, decision = (
        _with_concurrency(bundle, 2) for bundle in _bundles(root)
    )

    class BlockingRenderer(FakeRenderer):
        def __init__(self):
            super().__init__()
            self.started = Event()
            self.release = Event()

        def execute(self, bundle, **kwargs):
            self.started.set()
            self.release.wait(timeout=3)
            return super().execute(bundle, **kwargs)

    renderer = BlockingRenderer()
    executor = RenderBatchExecutorV1(
        renderer,
        maximum_concurrency=2,
        maximum_render_slots=2,
    )
    first_batch = RenderBatchV1.create(((flagship, "flagship"),))
    second_batch = RenderBatchV1.create(((decision, "decision"),))
    cancellation = Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            executor.execute_batch,
            first_batch,
            bundles={flagship.bundle_sha256: flagship},
            output_root=tmp_path / "first",
        )
        assert renderer.started.wait(timeout=2)
        second = pool.submit(
            executor.execute_batch,
            second_batch,
            bundles={decision.bundle_sha256: decision},
            output_root=tmp_path / "second",
            cancel_event=cancellation,
        )
        cancellation.set()
        assert second.result(timeout=2).status == "cancelled"
        assert renderer.calls == []
        renderer.release.set()
        assert first.result(timeout=3).status == "completed"
    assert renderer.calls == ["flagship"]


def test_render_batch_reexecutes_tampered_output(root: Path, tmp_path: Path) -> None:
    flagship, _ = _bundles(root)
    bundles = {flagship.bundle_sha256: flagship}
    batch = RenderBatchV1.create(((flagship, "flagship"),))
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=1)
    first = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    directory = tmp_path / f"batch-{batch.sha256[:16]}"
    (directory / "flagship.mp4").write_bytes(b"tampered")
    second = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    assert renderer.calls == ["flagship", "flagship"]
    assert first.to_dict() == second.to_dict()


def test_render_batch_requires_requested_playback_evidence(root: Path, tmp_path: Path) -> None:
    flagship, _ = _bundles(root)
    batch = RenderBatchV1.create(((flagship, "flagship"),))
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=1)
    bundles = {flagship.bundle_sha256: flagship}

    executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    executor.execute_batch(
        batch,
        bundles=bundles,
        output_root=tmp_path,
        playback_evidence=True,
    )

    assert renderer.playback_flags == [False, True]


def test_clip_manifest_reuse_requires_outcome_phase_check(root: Path, tmp_path: Path) -> None:
    _, decision = _bundles(root)
    batch = RenderBatchV1.create(((decision, "decision"),))
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=1)
    bundles = {decision.bundle_sha256: decision}
    assert executor.execute_batch(batch, bundles=bundles, output_root=tmp_path,
                                  playback_evidence=True).status == "completed"
    directory = tmp_path / f"batch-{batch.sha256[:16]}"
    item = batch.items[0]
    assert executor._existing_manifest(directory, item, playback_evidence=True, outcome_required=True) is not None
    manifest = directory / "decision-manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    outcome = value["playbackEvidence"].pop()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert executor._existing_manifest(directory, item, playback_evidence=True, outcome_required=True) is None
    assert executor._existing_manifest(directory, item, playback_evidence=True) is not None
    value["playbackEvidence"].append(outcome)
    value["playbackEvidence"][0]["checkpointPhase"] = "question"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert executor._existing_manifest(directory, item, playback_evidence=True) is None
    value["playbackEvidence"] = [{"completedPlayCount": count} for count in range(33)]
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert executor._existing_manifest(directory, item, playback_evidence=True) is None


def test_full_manifest_keeps_legacy_reads_but_new_renders_require_outcome(
    root: Path, tmp_path: Path,
) -> None:
    flagship, _ = _bundles(root)
    batch = RenderBatchV1.create(((flagship, "flagship"),))
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=1)
    bundles = {flagship.bundle_sha256: flagship}
    assert executor.execute_batch(batch, bundles=bundles, output_root=tmp_path,
                                  playback_evidence=True).status == "completed"
    directory = tmp_path / f"batch-{batch.sha256[:16]}"
    item = batch.items[0]
    manifest = directory / "flagship-manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert executor._existing_manifest(directory, item, playback_evidence=True, outcome_required=True) is not None
    outcome = value["playbackEvidence"].pop()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert executor._existing_manifest(directory, item, playback_evidence=True) is not None
    assert executor._existing_manifest(directory, item, playback_evidence=True, outcome_required=True) is None
    renderer.calls.clear()
    assert executor.execute_batch(batch, bundles=bundles, output_root=tmp_path,
                                  playback_evidence=True).status == "completed"
    assert renderer.calls == []
    value["playbackEvidence"].append({**outcome, "checkpointPhase": "continuation"})
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert executor._existing_manifest(directory, item, playback_evidence=True) is None

    class MissingOutcomeRenderer(FakeRenderer):
        def execute(self, bundle, **kwargs):
            artifacts = super().execute(bundle, **kwargs)
            path = Path(artifacts.manifest)
            evidence = json.loads(path.read_text(encoding="utf-8"))
            evidence["playbackEvidence"].pop()
            path.write_text(json.dumps(evidence), encoding="utf-8")
            return artifacts

    result = RenderBatchExecutorV1(MissingOutcomeRenderer(), maximum_concurrency=1).execute_batch(
        batch, bundles=bundles, output_root=tmp_path / "new", playback_evidence=True,
    )
    assert result.status == "failed"
    assert result.items[0].error_detail == "renderer output manifest failed idempotency verification"


def test_render_batch_keeps_full_proof_artifact_in_internal_receipt(
    root: Path,
    tmp_path: Path,
) -> None:
    flagship, _ = _bundles(root)
    proof_artifact_sha256 = "a" * 64
    batch = RenderBatchV1.create(
        ((flagship, "proof-bound"),),
        proof_artifact_hashes={
            flagship.bundle_sha256: proof_artifact_sha256,
        },
    )
    assert batch.items[0].proof_artifact_sha256 == proof_artifact_sha256
    assert batch.to_dict()["items"][0]["proofArtifactSha256"] == proof_artifact_sha256
    public_only_batch = RenderBatchV1.create(((flagship, "proof-bound"),))
    assert public_only_batch.sha256 != batch.sha256
    renderer = FakeRenderer()
    executor = RenderBatchExecutorV1(renderer, maximum_concurrency=1)
    bundles = {flagship.bundle_sha256: flagship}
    first = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    directory = tmp_path / f"batch-{batch.sha256[:16]}"
    manifest = directory / "proof-bound-manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert "proofArtifactSha256" not in value
    assert proof_artifact_sha256 not in manifest.read_text(encoding="utf-8")
    assert first.batch_sha256 == batch.sha256
    second = executor.execute_batch(batch, bundles=bundles, output_root=tmp_path)
    assert second.to_dict() == first.to_dict()
    assert renderer.calls == ["proof-bound"]


def test_render_batch_cancellation_and_failure_never_report_completion(root: Path, tmp_path: Path) -> None:
    flagship, decision = _bundles(root)
    bundles = {item.bundle_sha256: item for item in (flagship, decision)}
    batch = RenderBatchV1.create(((flagship, "flagship"), (decision, "decision")))
    cancel = Event()
    cancel.set()
    cancelled_renderer = FakeRenderer()
    cancelled = RenderBatchExecutorV1(cancelled_renderer, maximum_concurrency=2).execute_batch(
        batch, bundles=bundles, output_root=tmp_path / "cancelled", cancel_event=cancel,
    )
    assert cancelled.verify() and cancelled.status == "cancelled"
    assert {item.status for item in cancelled.items} == {"cancelled"}
    assert cancelled_renderer.calls == []

    failed_renderer = FakeRenderer(fail_basename="decision")
    failed = RenderBatchExecutorV1(failed_renderer, maximum_concurrency=2).execute_batch(
        batch, bundles=bundles, output_root=tmp_path / "failed",
    )
    assert failed.verify() and failed.status == "failed"
    assert [item.status for item in failed.items] == ["completed", "failed"]
    assert failed.items[1].error_code == "ValueError"
    assert failed.items[1].error_detail == "governed fake failure"


def test_render_batch_rejects_unknown_bundle_and_unsafe_concurrency(root: Path, tmp_path: Path) -> None:
    flagship, _ = _bundles(root)
    batch = RenderBatchV1.create(((flagship, "flagship"),))
    executor = RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=1)
    try:
        executor.execute_batch(batch, bundles={}, output_root=tmp_path)
    except ValueError as exc:
        assert "verification" in str(exc)
    else:
        raise AssertionError("missing bundle must fail closed")
    try:
        executor.execute_batch(
            batch,
            bundles={flagship.bundle_sha256: flagship},
            output_root=tmp_path,
            playback_evidence=1,
        )
    except TypeError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("coercive playback evidence flag must fail closed")
    for value in (0, 17):
        try:
            RenderBatchExecutorV1(FakeRenderer(), maximum_concurrency=value)
        except ValueError as exc:
            assert "concurrency" in str(exc)
        else:
            raise AssertionError("unsafe concurrency must fail closed")

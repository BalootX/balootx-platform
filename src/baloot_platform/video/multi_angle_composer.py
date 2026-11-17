"""Observer-isolated Remotion composition for governed multi-angle videos."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from baloot_platform.audit.models import content_hash
from baloot_platform.plugins.base import PluginDescriptor, RenderArtifacts

from .render_bundle import RenderBundleV1, _export_render
from .render_plan import MultiAngleRenderPlanV1
from .remotion_renderer import RemotionProfessionalRenderer, _json_write, _sha256, _timestamp
from .media_validation import media_tool, probe_media, validate_media_probe

# Stable test seam; implementation authority remains media_validation.py.
_probe_media = probe_media


def _canonical_manifest_sha(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("manifestSha256", None)
    return hashlib.sha256(
        json.dumps(
            core, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


class MultiAngleComposerV1(RemotionProfessionalRenderer):
    """Compose verified perspective videos without joining their scene data."""

    descriptor = PluginDescriptor(
        id="multi_angle_composer_v1",
        kind="video_compositor",
        version="1.0.3",
        description="Observer-isolated composition of verified Remotion perspective renders",
        capabilities=(
            "multi_angle_render_plan_v1",
            "separate_render_bundles",
            "explicit_perspective_boundaries",
            "cut_transition",
            "fade_through_black_transition",
            "captions",
            "decode_validation",
            "1080x1920",
            "30fps",
            "60fps",
        ),
        resource_class="video-cpu-premium",
        deterministic=True,
        isolation="process",
        side_effects=("filesystem_output",),
        timeout_seconds=900,
        maximum_memory_mb=4096,
        compatible_profile_schemas=("1",),
    )

    def __init__(self, video_profile: Mapping[str, Any], *, repository_root: str | Path | None = None):
        super().__init__(video_profile, repository_root=repository_root)
        if not (self.renderer_root / "scripts/render-multi-angle.mjs").is_file():
            raise FileNotFoundError("Remotion multi-angle composition script is missing")
        self.ffprobe: Path | None = None

    def _ffprobe(self) -> Path:
        if self.ffprobe is None:
            self.ffprobe = media_tool(self.repository_root, "ffprobe")
        return self.ffprobe

    def _source(
        self,
        *,
        manifest_path: Path,
        bundle: RenderBundleV1,
    ) -> tuple[Path, dict[str, Any]]:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            value.get("manifestSha256") != _canonical_manifest_sha(value)
            or value.get("bundleSha256") != bundle.bundle_sha256
            or value.get("observerSafe") is not True
        ):
            raise ValueError("multi-angle source manifest identity or observer-safety check failed")
        video_record = dict((value.get("files") or {}).get("video") or {})
        video = (manifest_path.parent / str(video_record.get("name") or "")).resolve()
        if video.parent != manifest_path.parent.resolve() or not video.is_file():
            raise ValueError("multi-angle source video is outside its governed manifest directory")
        if (
            video.stat().st_size != int(video_record.get("bytes", -1))
            or _sha256(video) != video_record.get("sha256")
        ):
            raise ValueError("multi-angle source video hash or size check failed")
        evidence = {
            "bundleSha256": bundle.bundle_sha256,
            "observerSeat": bundle.value["observerSeat"],
            "manifestSha256": value["manifestSha256"],
            "videoSha256": video_record["sha256"],
            "videoBytes": video_record["bytes"],
        }
        return video, evidence

    def _existing(
        self,
        *,
        manifest: Path,
        plan: MultiAngleRenderPlanV1,
        composition_sha: str,
        source_identities: tuple[dict[str, Any], ...],
    ) -> RenderArtifacts | None:
        if not manifest.is_file():
            return None
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if not (
                value.get("manifestSha256") == _canonical_manifest_sha(value)
                and value.get("renderer") == self.descriptor.to_dict()
                and value.get("planSha256") == plan.sha256
                and value.get("compositionSha256") == composition_sha
                and value.get("sourceIdentities") == list(source_identities)
                and value.get("observerSafe") is True
                and value.get("observerIsolation") == "separately_rendered_media_only"
            ):
                return None
            files = value.get("files")
            if not isinstance(files, Mapping):
                return None
            resolved: dict[str, Path] = {}
            for key, raw in files.items():
                if not isinstance(raw, Mapping):
                    return None
                path = (manifest.parent / str(raw.get("name") or "")).resolve()
                if (
                    path.parent != manifest.parent.resolve()
                    or not path.is_file()
                    or path.stat().st_size != int(raw.get("bytes", -1))
                    or _sha256(path) != raw.get("sha256")
                ):
                    return None
                resolved[str(key)] = path
            required = {
                "video", "poster", "plan", "captionsSrt", "captionsVtt",
                "transcript", "contactSheet",
            }
            if not required.issubset(resolved):
                return None
            validate_media_probe(
                value["mediaProbe"],
                width=plan.width,
                height=plan.height,
                fps=plan.fps,
                minimum_frames=plan.total_frames,
                expected_frames=plan.total_frames,
                audio_rate=int(self.profile["render"]["audio_rate"]),
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return RenderArtifacts(
            video=resolved["video"],
            poster=resolved["poster"],
            episode_json=resolved["plan"],
            captions=resolved["captionsSrt"],
            contact_sheet=resolved["contactSheet"],
            manifest=manifest,
            timeline=resolved["plan"],
            captions_vtt=resolved["captionsVtt"],
            transcript=resolved["transcript"],
        )

    @staticmethod
    def _caption_documents(
        plan: MultiAngleRenderPlanV1,
        bundles: Mapping[str, RenderBundleV1],
    ) -> tuple[str, str, str]:
        cues: list[tuple[int, int, str]] = []
        output_start = 0
        for segment in plan.segments:
            bundle = bundles[segment.bundle_sha256]
            reveal = next(
                (
                    item for item in bundle.value["timeline"]["segments"]
                    if item["stage"] == "reveal"
                ),
                None,
            )
            if reveal is not None:
                reveal_start = int(reveal["startFrame"])
                reveal_end = reveal_start + int(reveal["durationFrames"])
                overlap_start = max(segment.source_start_frame, reveal_start)
                overlap_end = min(segment.source_end_frame, reveal_end)
                reason = MultiAngleComposerV1._reveal_reason(bundle)
                if overlap_end > overlap_start and reason:
                    cues.append((
                        output_start + overlap_start - segment.source_start_frame,
                        output_start + overlap_end - segment.source_start_frame,
                        reason,
                    ))
            output_start += segment.source_end_frame - segment.source_start_frame
        vtt = ["WEBVTT", ""]
        srt: list[str] = []
        transcript: list[str] = []
        for index, (start, end, reason) in enumerate(cues, start=1):
            vtt.extend([
                f"{_timestamp(start, plan.fps)} --> {_timestamp(end, plan.fps)}",
                reason,
                "",
            ])
            srt.extend([
                str(index),
                f"{_timestamp(start, plan.fps, srt=True)} --> {_timestamp(end, plan.fps, srt=True)}",
                reason,
                "",
            ])
            if reason not in transcript:
                transcript.append(reason)
        return "\n".join(vtt), "\n".join(srt), "\n".join(transcript) + ("\n" if transcript else "")

    def execute(self, request: Any, **kwargs: Any) -> RenderArtifacts:
        if not isinstance(request, MultiAngleRenderPlanV1):
            raise TypeError("multi-angle composer requires MultiAngleRenderPlanV1")
        bundles = kwargs.get("bundles")
        manifests = kwargs.get("source_manifests")
        if not isinstance(bundles, Mapping) or not request.verify(bundles):
            raise ValueError("multi-angle plan or render bundles failed verification")
        if not isinstance(manifests, Mapping):
            raise ValueError("multi-angle source manifests are required")
        profile_render = self.profile["render"]
        for segment in request.segments:
            render = _export_render(
                self.profile, bundles[segment.bundle_sha256].value["exportPreset"]
            )
            if (request.width, request.height, request.fps) != (
                int(render["width"]), int(render["height"]), int(render["fps"])
            ):
                raise ValueError("multi-angle plan does not match the governed video profile")

        output_dir = Path(kwargs.get("output_dir", "outputs")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = str(kwargs.get("basename") or f"multi-angle-{request.sha256[:12]}")
        if not basename or any(character in basename for character in "\\/:*?\"<>|"):
            raise ValueError("multi-angle basename is invalid")

        unique_sources: dict[str, tuple[Path, dict[str, Any]]] = {}
        for segment in request.segments:
            if segment.bundle_sha256 in unique_sources:
                continue
            raw_manifest = manifests.get(segment.bundle_sha256)
            if raw_manifest is None:
                raise ValueError("multi-angle source manifest is missing")
            unique_sources[segment.bundle_sha256] = self._source(
                manifest_path=Path(raw_manifest).resolve(),
                bundle=bundles[segment.bundle_sha256],
            )

        source_identities: tuple[dict[str, Any], ...] = tuple(
            item[1] for item in unique_sources.values()
        )
        composition_core = {
            "schemaVersion": 1,
            "planSha256": request.sha256,
            "observerIsolation": "separately_rendered_media_only",
            "sourceIdentities": list(source_identities),
        }
        composition_sha = content_hash(composition_core)
        manifest = output_dir / f"{basename}-manifest.json"
        existing = self._existing(
            manifest=manifest,
            plan=request,
            composition_sha=composition_sha,
            source_identities=source_identities,
        )
        if existing is not None:
            return existing

        source_media: list[dict[str, Any]] = []
        for bundle_sha, (video, identity) in unique_sources.items():
            bundle = bundles[bundle_sha]
            maximum = max(
                item.source_end_frame
                for item in request.segments
                if item.bundle_sha256 == bundle_sha
            )
            probe = _probe_media(video, ffprobe=self._ffprobe())
            validate_media_probe(
                probe,
                width=int(bundle.value["render"]["width"]),
                height=int(bundle.value["render"]["height"]),
                fps=int(bundle.value["render"]["fps"]),
                minimum_frames=maximum,
                audio_rate=int(bundle.value["render"]["audio_rate"]),
            )
            source_media.append({**identity, "mediaProbe": probe})

        with tempfile.TemporaryDirectory(prefix=".baloot-multi-angle-", dir=output_dir) as temporary:
            work = Path(temporary)
            public = work / "public"
            public.mkdir()
            source_names: dict[str, str] = {}
            for index, (bundle_sha, (video, _)) in enumerate(unique_sources.items()):
                name = f"source-{index:02d}.mp4"
                shutil.copyfile(video, public / name)
                source_names[bundle_sha] = name
            segments = []
            for segment in request.segments:
                bundle = bundles[segment.bundle_sha256]
                players = bundle.value["identity"]["players"]
                value = segment.to_dict()
                value.update({
                    "sourceName": source_names[segment.bundle_sha256],
                    "perspectiveName": players[segment.observer_seat],
                })
                segments.append(value)
            input_value = {
                "plan": {
                    "sha256": request.sha256,
                    "fps": request.fps,
                    "width": request.width,
                    "height": request.height,
                    "totalFrames": request.total_frames,
                    "segments": segments,
                },
                "render": {
                    "crf": int(profile_render["crf"]),
                    "concurrency": int(profile_render["concurrency"]),
                },
            }
            input_path = work / "multi-angle-input.json"
            _json_write(input_path, input_value)
            self._run_renderer(
                script=self.renderer_root / "scripts/render-multi-angle.mjs",
                environment={
                    "BALOOTX_MULTI_ANGLE_INPUT": str(input_path),
                    "BALOOTX_MULTI_ANGLE_PUBLIC": str(public),
                    "BALOOTX_RENDER_OUTPUT": str(work),
                    "BALOOTX_RENDER_BASENAME": "render",
                },
                cancel_event=kwargs.get("cancel_event"),
            )
            video = output_dir / f"{basename}.mp4"
            shutil.copyfile(work / "render.mp4", video)
            checkpoints: list[Path] = []
            for source in sorted(work.glob("checkpoint-*.png")):
                destination = output_dir / f"{basename}-{source.name}"
                shutil.copyfile(source, destination)
                checkpoints.append(destination)
            if len(checkpoints) != len(request.segments):
                raise RuntimeError("multi-angle renderer produced incomplete visual checkpoints")

        probe = _probe_media(video, ffprobe=self._ffprobe())
        validate_media_probe(
            probe,
            width=request.width,
            height=request.height,
            fps=request.fps,
            minimum_frames=request.total_frames,
            expected_frames=request.total_frames,
            audio_rate=int(profile_render["audio_rate"]),
        )
        poster = output_dir / f"{basename}.jpg"
        with Image.open(checkpoints[0]) as source:
            source.convert("RGB").save(
                poster, format="JPEG", quality=95, subsampling=0, optimize=False
            )
        contact = output_dir / f"{basename}-contact.jpg"
        self._contact_sheet(checkpoints, contact)

        captions_vtt = output_dir / f"{basename}.vtt"
        captions_srt = output_dir / f"{basename}.srt"
        transcript = output_dir / f"{basename}-transcript.txt"
        vtt, srt, text = self._caption_documents(request, bundles)
        captions_vtt.write_text(vtt, encoding="utf-8", newline="\n")
        captions_srt.write_text(srt, encoding="utf-8", newline="\n")
        transcript.write_text(text, encoding="utf-8", newline="\n")

        plan_path = output_dir / f"{basename}.json"
        _json_write(plan_path, {**composition_core, "compositionSha256": composition_sha})
        files = {
            "video": video,
            "poster": poster,
            "plan": plan_path,
            "captionsSrt": captions_srt,
            "captionsVtt": captions_vtt,
            "transcript": transcript,
            "contactSheet": contact,
            **{f"checkpoint:{index}": path for index, path in enumerate(checkpoints)},
        }
        manifest_value = {
            "schemaVersion": 1,
            "renderer": self.descriptor.to_dict(),
            "planSha256": request.sha256,
            "compositionSha256": composition_sha,
            "observerSafe": True,
            "observerIsolation": "separately_rendered_media_only",
            "deterministic": True,
            "sourceIdentities": list(source_identities),
            "sourceMedia": source_media,
            "mediaProbe": probe,
            "files": {
                key: {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for key, path in sorted(files.items())
            },
        }
        manifest_value["manifestSha256"] = _canonical_manifest_sha(manifest_value)
        _json_write(manifest, manifest_value)
        return RenderArtifacts(
            video=video,
            poster=poster,
            episode_json=plan_path,
            captions=captions_srt,
            contact_sheet=contact,
            manifest=manifest,
            timeline=plan_path,
            captions_vtt=captions_vtt,
            transcript=transcript,
        )

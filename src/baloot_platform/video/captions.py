"""Deterministic caption derivatives from one SRT authority."""
from __future__ import annotations

from pathlib import Path


def derive_caption_files(
    srt_path: Path,
    vtt_path: Path,
    transcript_path: Path,
) -> None:
    lines = srt_path.read_text(encoding="utf-8").splitlines()
    vtt_lines = ["WEBVTT", ""]
    transcript_lines: list[str] = []
    for line in lines:
        vtt_lines.append(line.replace(",", ".") if "-->" in line else line)
        stripped = line.strip()
        if (
            stripped
            and not stripped.isdigit()
            and "-->" not in stripped
            and (not transcript_lines or transcript_lines[-1] != stripped)
        ):
            transcript_lines.append(stripped)
    vtt_path.write_text(
        "\n".join(vtt_lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    transcript_path.write_text(
        "\n".join(transcript_lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )

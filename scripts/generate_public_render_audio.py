"""Generate fresh public speech with the system eSpeak NG Arabic voice; no recordings."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[1]
TERMS_URL = "https://github.com/espeak-ng/espeak-ng/blob/1.51/COPYING"
VOICE_TEXT = {
    "pass": "????", "sun": "????", "hokum": "??????",
    "hokum_second": "?????? ??????", "ashkal": "????????", "wala": "?????",
    "double": "??????", "triple": "?????", "four": "?????", "gahwa": "????????",
    "siri": "?????", "fifty": "?????????", "hundred": "???????",
    "four_hundred": "???????? ???????", "baloot": "???????",
}


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, timeout=30, **kwargs)


def generate(root: Path = ROOT) -> None:
    engine = shutil.which("espeak-ng")
    ffmpeg = shutil.which("ffmpeg")
    if not engine or not ffmpeg:
        raise RuntimeError("install espeak-ng and ffmpeg before generating public render audio")
    version = _run([engine, "--version"], text=True).stdout.split("Data at:")[0].strip()
    if not version.startswith("eSpeak NG text-to-speech: 1.51"):
        raise RuntimeError("public speech terms are bound to eSpeak NG 1.51")
    base = root / "assets/audio/voice_packs"
    first = json.loads((base / "ar_sa_baloot_v1/generated-manifest.json").read_text(encoding="utf-8"))
    second = json.loads((base / "ar_sa_baloot_v2_smooth/manifest.json").read_text(encoding="utf-8"))
    texts = {key: item["textAr"] for key, item in first["clips"].items()}
    if texts.keys() != VOICE_TEXT.keys():
        raise ValueError("public speech text inventory mismatch")
    texts["explanation"] = second["clips"]["explanation"]["textAr"]
    records = {}
    with tempfile.TemporaryDirectory(prefix="baloot-public-speech-") as temporary:
        stage = Path(temporary)
        for key, text in texts.items():
            raw, output = stage / "raw.wav", stage / f"{key}.wav"
            synthesis_text = VOICE_TEXT.get(key, text)
            _run([engine, "-v", "ar", "-s", "180", "-w", str(raw), "--stdin"], input=synthesis_text.encode("utf-8"))
            _run([ffmpeg, "-v", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(output)])
            with wave.open(str(output), "rb") as stream:
                if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (48000, 1, 2) or stream.getnframes() == 0:
                    raise ValueError(f"invalid generated speech: {key}")
                duration = stream.getnframes() / stream.getframerate()
            records[key] = {"file": output.name, "textAr": text, "synthesisText": synthesis_text,
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "durationSeconds": duration, "sampleRate": 48000, "channels": 1}
        for pack, filename, keys in [
            ("ar_sa_baloot_v1", "generated-manifest.json", VOICE_TEXT),
            ("ar_sa_baloot_v2_smooth", "manifest.json", texts),
        ]:
            directory = base / pack
            directory.mkdir(parents=True, exist_ok=True)
            for key in keys:
                shutil.copyfile(stage / f"{key}.wav", directory / f"{key}.wav")
            manifest = {"schemaVersion": 2, "packId": pack, "provider": "espeak-ng-ar-public",
                        "generatorVersion": version, "voice": "ar", "speed": 180,
                        "generatorSha256": hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
                        "termsUrl": TERMS_URL, "copiedReferenceAudio": False,
                        "provenance": "fresh synthesis from public project text; no legacy WAV reuse",
                        "clips": {key: records[key] for key in keys}}
            (directory / filename).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    generate()

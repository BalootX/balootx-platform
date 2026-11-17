from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from baloot_platform.config import platform_root


@dataclass(frozen=True)
class CardPackDefinition:
    pack_id: str
    label_ar: str
    provider: str
    backs: tuple[str, ...]
    default_back: str
    enabled: bool
    freeze_in_shared_environment: bool
    license_status: str
    intended_use: str
    manifest: str | None = None

    def validate_back(self, back_id: str) -> str:
        selected = str(back_id or self.default_back)
        if selected not in self.backs:
            raise ValueError(f"unknown card back {selected!r} for pack {self.pack_id}")
        return selected

    def profile_fragment(self, back_id: str | None = None) -> dict[str, Any]:
        selected_back = self.validate_back(back_id or self.default_back)
        value: dict[str, Any] = {
            "provider": self.provider,
            "pack_id": self.pack_id,
            "back_id": selected_back,
            "freeze_in_shared_environment": self.freeze_in_shared_environment,
            "license_status": self.license_status,
            "intended_use": self.intended_use,
        }
        if self.manifest:
            value["manifest"] = self.manifest
        if self.provider != "procedural_v1":
            value["fallback_provider"] = "procedural_v1"
        return value

    def option(self) -> dict[str, Any]:
        return {
            "id": self.pack_id,
            "labelAr": self.label_ar,
            "provider": self.provider,
            "backs": list(self.backs),
            "defaultBack": self.default_back,
            "licenseStatus": self.license_status,
            "intendedUse": self.intended_use,
        }


class CardPackRegistry:
    """Validated, versionable registry for renderer card assets.

    The registry is presentation-only.  It cannot alter card identity or any
    Baloot rule.  It exists so tenants, rooms, shared games and video profiles
    can select an asset pack without hard-coded renderer branches.
    """

    def __init__(self, definitions: dict[str, CardPackDefinition], *, default_pack: str, root: Path) -> None:
        if default_pack not in definitions:
            raise ValueError(f"default card pack is not registered: {default_pack}")
        self._definitions = dict(definitions)
        self.default_pack = default_pack
        self.root = root
        self._validate_assets()

    @classmethod
    def from_yaml(cls, path: str | Path, *, root: str | Path | None = None) -> "CardPackRegistry":
        resolved_root = platform_root(root)
        source = Path(path)
        if not source.is_absolute():
            source = resolved_root / source
        data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        definitions: dict[str, CardPackDefinition] = {}
        for pack_id, raw in dict(data.get("packs") or {}).items():
            value = dict(raw or {})
            definition = CardPackDefinition(
                pack_id=str(pack_id),
                label_ar=str(value.get("label_ar") or pack_id),
                provider=str(value.get("provider") or "procedural_v1"),
                backs=tuple(str(item) for item in (value.get("backs") or ["1B"])),
                default_back=str(value.get("default_back") or "1B"),
                enabled=bool(value.get("enabled", True)),
                freeze_in_shared_environment=bool(value.get("freeze_in_shared_environment", True)),
                license_status=str(value.get("license_status") or "unknown"),
                intended_use=str(value.get("intended_use") or "unspecified"),
                manifest=str(value["manifest"]) if value.get("manifest") else None,
            )
            if definition.enabled:
                definitions[definition.pack_id] = definition
        return cls(definitions, default_pack=str(data.get("default_pack") or ""), root=resolved_root)

    def _validate_assets(self) -> None:
        for definition in self._definitions.values():
            definition.validate_back(definition.default_back)
            if definition.provider == "raster_pack_v1":
                if not definition.manifest:
                    raise ValueError(f"raster pack {definition.pack_id} requires a manifest")
                path = self.root / definition.manifest
                if not path.is_file():
                    raise FileNotFoundError(f"card-pack manifest not found: {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                if str(data.get("packId")) != definition.pack_id:
                    raise ValueError(f"card-pack manifest identity mismatch: {definition.pack_id}")
                cards = dict(data.get("cards") or {})
                if len(cards) != 32:
                    raise ValueError(f"card pack {definition.pack_id} must contain exactly 32 Baloot cards")
                backs = dict(data.get("backs") or {})
                for back_id in definition.backs:
                    if back_id not in backs:
                        raise ValueError(f"card back {back_id} missing from {definition.pack_id}")

    def get(self, pack_id: str | None = None) -> CardPackDefinition:
        selected = str(pack_id or self.default_pack)
        try:
            return self._definitions[selected]
        except KeyError as exc:
            raise ValueError(f"unknown or disabled card pack: {selected}") from exc

    def profile_fragment(self, pack_id: str | None = None, back_id: str | None = None) -> dict[str, Any]:
        return copy.deepcopy(self.get(pack_id).profile_fragment(back_id))

    def options(self) -> list[dict[str, Any]]:
        values = [definition.option() for definition in self._definitions.values()]
        values.sort(key=lambda item: (item["id"] != self.default_pack, item["id"]))
        return values

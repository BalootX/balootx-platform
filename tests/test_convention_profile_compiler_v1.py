from __future__ import annotations

import copy
import importlib.util
import json

import pytest
import yaml

from baloot_platform.config import build_bundle
from baloot_platform.research import convention_proofs
from baloot_platform.research.convention_proofs import (
    convention_behavioral_proofs,
    validate_convention_profile,
)


def test_convention_profile_has_complete_truthful_leaf_proofs(root):
    bundle = build_bundle(root)
    proofs = convention_behavioral_proofs(
        conventions=bundle.convention,
        rules=bundle.rules.data,
    )
    def leaf_ids(value, prefix=("conventions",)):
        return {
            item
            for key, child in value.items()
            for item in (
                leaf_ids(child, (*prefix, key))
                if isinstance(child, dict)
                else {".".join((*prefix, key))}
            )
        }

    assert set(proofs) == leaf_ids(bundle.convention)
    assert {
        key for key, proof in proofs.items() if proof["status"] == "proven_invariant"
    } == {
        "conventions.profile.id",
        "conventions.profile.status",
        "conventions.profile.version",
        "conventions.schema_version",
    }
    behavioral = [
        proof for proof in proofs.values() if proof["status"] == "proven_behavioral"
    ]
    assert len(behavioral) == len(proofs) - 4
    assert all(
        proof["proofKind"]
        == "observer_safe_detector_fixture_likelihood_counterfactual"
        and proof["observerProjectionSafe"] is True
        and proof["softEvidenceOnly"] is True
        and proof["candidateStateMutation"] is False
        and proof["legalCandidateSetChanged"] is False
        and proof["productionDetector"] == "analysis.signals.detect_signals"
        and len(proof["observationHash"]) == 64
        and len(proof["detectorFixtureStateHashes"]) == 2
        and proof["observationKind"]
        not in proof["counterexampleObservationKinds"]
        and proof["baselineWeight"] != proof["counterfactualWeight"]
        for proof in behavioral
    )


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("schema_version",), 2),
        (("profile", "id"), "INVALID ID"),
        (("profile", "version"), "one"),
        (("profile", "status"), "unknown"),
        (("forcedness", "near_forced"), -0.1),
        (("relationship_reliability", "partner"), 1.1),
        (
            (
                "signals",
                "minimum_winner",
                "likelihood_ratios",
                "preserve_strength",
            ),
            4.1,
        ),
    ),
)
def test_convention_profile_rejects_invalid_values(root, path, value):
    profile = copy.deepcopy(build_bundle(root).convention)
    cursor = profile
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(ValueError):
        validate_convention_profile(profile)


def test_convention_forcedness_must_remain_monotonic(root):
    profile = copy.deepcopy(build_bundle(root).convention)
    profile["forcedness"]["near_forced"] = 0.6
    with pytest.raises(ValueError, match="monotonic"):
        validate_convention_profile(profile)


def test_only_governed_convention_ratios_are_authored(root):
    profile = build_bundle(root).convention
    assert set(profile["signals"]) == {
        "high_expenditure",
        "high_then_low",
        "low_then_high",
        "minimum_winner",
        "partner_return",
    }
    assert "preserve_high_card" not in profile["signals"]["low_then_high"][
        "likelihood_ratios"
    ]
    assert "no_signal_or_deception" not in profile["signals"]["high_then_low"][
        "likelihood_ratios"
    ]


def test_enabled_detector_completeness_gate_is_dynamic(root, monkeypatch):
    profile = build_bundle(root).convention
    monkeypatch.setattr(
        convention_proofs,
        "PRODUCTION_SIGNAL_KINDS",
        frozenset({"high_then_low", "low_then_high", "partner_return"}),
    )
    with pytest.raises(ValueError, match="lack production weighting"):
        convention_behavioral_proofs(
            conventions=profile,
            rules=build_bundle(root).rules.data,
        )


def test_profile_schema_regeneration_preserves_v1_protocol_invariants(root):
    generator_path = root / "scripts/generate_profile_schemas.py"
    spec = importlib.util.spec_from_file_location(
        "generate_profile_schemas",
        generator_path,
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    bundle = yaml.safe_load((root / "configs/bundle.yaml").read_text(encoding="utf-8"))
    checked = []
    for family, relative in sorted(bundle["components"].items()):
        payload = yaml.safe_load(
            (root / "configs" / relative).read_text(encoding="utf-8")
        )
        if "schema_version" not in payload:
            continue
        generated = generator.infer(payload)
        generator.apply_release_constraints(family, generated, payload)
        expected = {
            "const": payload["schema_version"],
            "type": "integer",
        }
        assert generated["properties"]["schema_version"] == expected
        authored_schema = json.loads(
            (root / "schemas/profiles" / f"{family}.schema.json").read_text(
                encoding="utf-8"
            )
        )
        packaged_schema = json.loads(
            (
                root
                / "src/baloot_platform/resources/schemas/profiles"
                / f"{family}.schema.json"
            ).read_text(encoding="utf-8")
        )
        assert authored_schema["properties"]["schema_version"] == expected
        assert packaged_schema["properties"]["schema_version"] == expected
        checked.append(family)

    assert "conventions" in checked
    assert "product" in checked
    assert "ml" in checked
    assert "match_value" in checked
    assert json.loads(
        (root / "schemas/profiles/conventions.schema.json").read_text(
            encoding="utf-8"
        )
    )["properties"]["schema_version"]["const"] == 1
    assert json.loads(
        (root / "schemas/profiles/ml.schema.json").read_text(encoding="utf-8")
    )["properties"]["schema_version"]["const"] == 2
    convention_schema = json.loads(
        (root / "schemas/profiles/conventions.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert convention_schema["properties"]["relationship_reliability"][
        "properties"
    ]["partner"] == {"maximum": 1.0, "minimum": 0.0, "type": "number"}
    assert set(
        convention_schema["properties"]["signals"]["properties"]
    ) == {
        "high_expenditure",
        "high_then_low",
        "low_then_high",
        "minimum_winner",
        "partner_return",
    }
    assert convention_schema["properties"]["signals"]["properties"][
        "minimum_winner"
    ]["properties"]["likelihood_ratios"]["properties"][
        "preserve_strength"
    ] == {"exclusiveMinimum": 0.0, "maximum": 4.0, "type": "number"}

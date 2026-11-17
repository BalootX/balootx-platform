import copy
import itertools
import json

import pytest

from scripts import render_research_notice as notice


NEW_IDS = tuple(f"sun-single-winner-20261118-{index:02d}-v1" for index in range(1, 5))


def _opener(normalized):
    return (
        tuple(sorted(card["code"] for card in normalized.state["hands"][normalized.next_actor])),
        tuple(sorted(play["card"]["code"] for play in normalized.state["currentTrick"]["plays"])),
    )


def _deal_signature(fixture, rules):
    prefix = {**fixture["exactTracePrefix"], "playActions": [],
              "projectDeclarations": [], "projectAssertions": [], "stateAssertions": {}}
    hands = notice.ExactStateNormalizerV1(rules).normalize(prefix).state["hands"]
    # Ignore all suit and seat renaming when checking whether a deal is new.
    return min(
        tuple(sorted(tuple(sorted(permutation[card["id"] // 8] * 8 + card["id"] % 8
                                  for card in hand)) for hand in hands.values()))
        for permutation in itertools.permutations(range(4))
    )


@pytest.mark.parametrize("identifier", NEW_IDS)
def test_new_catalog_deal_builds_exact_complete_scenes(identifier, rules):
    selection = notice.select_deal(identifier)
    fixture = json.loads(selection["fixture"].read_text(encoding="utf-8"))
    assert "?" not in fixture["caption"] and "?" not in fixture["titleAr"]
    assert "\ufffd" not in fixture["caption"] and "\ufffd" not in fixture["titleAr"]
    expected = fixture["expected"]
    profile = notice.notice_profile(public=True)
    profile["identity"] = json.loads((notice.ROOT / "examples/synthetic-clip-identity.json").read_text(encoding="utf-8"))
    names = profile["identity"]["player_name_pool"]
    assert len(set(names)) == 16
    assert all(1 <= len(name) <= 5 and all("\u0621" <= char <= "\u064a" for char in name) for name in names)
    notice.validate_approved_v4_public_profile(profile)
    bundle = notice.build_notice_bundle(
        rules, profile, fixture=selection["fixture"], golden=selection["golden"],
    )
    result = notice.engine_round_result(rules, selection["fixture"])
    notice.validate_notice_bundle(bundle, result, public=True, fixture=selection["fixture"])
    reveal = next(scene for scene in bundle.value["scenes"] if scene["stage"] == "reveal")
    analysis = next(layer["content"] for layer in reveal["layers"] if layer["id"] == "analysis")
    assert {row["action"]: row["value"] for row in analysis["rootActions"]} == expected["rootValues"]
    assert sorted(row["action"] for row in analysis["rootActions"]) == expected["legalRootActions"]
    assert {key: result["roundScore"][key] for key in expected["roundScore"]} == expected["roundScore"]
    plays = notice._lifecycle(bundle.value)["plays"]
    actual = [{"actor": play["seat"], "card": play["card"]["code"]} for play in plays]
    assert actual[:len(fixture["exactTracePrefix"]["playActions"])] == fixture["exactTracePrefix"]["playActions"]
    assert [play["card"] for play in actual[len(fixture["exactTracePrefix"]["playActions"]):]] == expected["principalVariation"]
    assert len({play["card"] for play in actual}) == 32
    assert selection["renderAdmission"]["viewer"] != selection["renderAdmission"]["buyer"]
    assert selection["caption"] == fixture["caption"]
    assert selection["scenarioFamily"] == fixture["scenarioFamily"]
    players = bundle.value["identity"]["players"]
    assert len(set(players.values())) == 4
    assert set(players.values()) <= set(profile["identity"]["player_name_pool"])
    assert all(not any(char.isdigit() for char in name) for name in players.values())


def test_new_deals_do_not_reuse_existing_hands_or_openers(rules):
    fixtures = [json.loads((notice.ROOT / relative).read_text(encoding="utf-8"))
                for _, relative, _ in notice.DEAL_SOURCES]
    fixtures.append(json.loads((notice.ROOT / "examples/research-scenario-v4-ace-sacrifice-seven.json").read_text(encoding="utf-8")))
    old = [fixture for fixture in fixtures if fixture["scenarioId"] not in NEW_IDS]
    fresh = [fixture for fixture in fixtures if fixture["scenarioId"] in NEW_IDS]
    normalizer = notice.ExactStateNormalizerV1(rules)
    seen_deals = {_deal_signature(fixture, rules) for fixture in old}
    seen_openers = {_opener(normalizer.normalize(fixture["exactTracePrefix"])) for fixture in old}
    seen_captions = {notice.CAPTION}
    families = []
    for fixture in fresh:
        deal = _deal_signature(fixture, rules)
        opener = _opener(normalizer.normalize(fixture["exactTracePrefix"]))
        assert deal not in seen_deals
        assert opener not in seen_openers
        assert fixture["caption"] not in seen_captions
        assert not families or fixture["scenarioFamily"] != families[-1]
        seen_deals.add(deal)
        seen_openers.add(opener)
        seen_captions.add(fixture["caption"])
        families.append(fixture["scenarioFamily"])
    assert len(fresh) == 4


@pytest.mark.parametrize("key", ["caption", "scenarioFamily"])
@pytest.mark.parametrize("value", ["", "  ", None, 1])
def test_catalog_content_metadata_rejects_invalid_text(tmp_path, monkeypatch, key, value):
    fixture = copy.deepcopy(json.loads(notice.select_deal(NEW_IDS[0])["fixture"].read_text(encoding="utf-8")))
    fixture[key] = value
    (tmp_path / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr(notice, "ROOT", tmp_path)
    monkeypatch.setattr(notice, "DEAL_SOURCES", ((NEW_IDS[0], "fixture.json", "new-deal"),))
    with pytest.raises(ValueError, match=f"{key} must be non-empty text"):
        notice.select_deal(NEW_IDS[0])

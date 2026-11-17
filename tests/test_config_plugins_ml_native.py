import copy
import json
import os
from pathlib import Path
import pytest

from baloot_platform.config import PROFILE_FAMILIES, build_bundle, write_profile_manifest
from baloot_platform.core.bots import BotProfile, HeuristicBot
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.enums import Contract,Seat,Suit
from baloot_platform.analysis.models import InformationState
from baloot_platform.ml.actions import ACTION_COUNT,ActionCodec,ActionKind
from baloot_platform.ml.features import FEATURE_COUNT,feature_names,encode_information_state
from baloot_platform.ml.environment import BalootEnvironment
from baloot_platform.native.accelerator import NativeAccelerator
from baloot_platform.plugins.base import PluginDescriptor
from baloot_platform.plugins.registry import PluginPolicy, PluginRegistry


def test_bundle_has_every_profile_family_and_stable_hash(root,tmp_path):
    a=build_bundle(root);b=build_bundle(root)
    assert a.sha256==b.sha256 and set(a.components)==set(PROFILE_FAMILIES)
    path=write_profile_manifest(root,tmp_path/'manifest.json')
    data=json.loads(path.read_text())
    assert set(data['bundles'])=={'full','sun_only','hokum_only'}
    assert data['bundles']['full']['sha256']==a.sha256


def test_mode_overlays_limit_contracts(root):
    assert build_bundle(root,mode='sun_only').rules.data['contracts']['allowed']==['sun']
    assert build_bundle(root,mode='hokum_only').rules.data['contracts']['allowed']==['hokum']
    with pytest.raises(ValueError):build_bundle(root,mode='unknown')


def test_plugin_allowlist_and_duplicate_rejected():
    class P: descriptor=PluginDescriptor('x','test','1.0','x')
    policy=PluginPolicy.from_profile({
        "host_api_version":"1","profile_schema_version":"1","allowed_ids":["x"],
        "allowed_isolation":["in_process"],"allowed_permissions":["none"],
        "require_signed_external":True,"require_content_addressed_external":True,
        "maximum_timeout_seconds":30,"maximum_response_bytes":1048576,
        "maximum_memory_mb":512,"maximum_cpu_seconds":10,
        "maximum_stderr_bytes":65536,
    })
    reg=PluginRegistry(policy=policy);reg.register(P())
    assert reg.get('test','x').descriptor.id=='x'
    with pytest.raises(ValueError):reg.register(P())
    class Y: descriptor=PluginDescriptor('y','test','1.0','y')
    with pytest.raises(PermissionError):reg.register(Y())


def test_action_codec_contract():
    assert ACTION_COUNT==76
    for card in range(32):
        assert ActionCodec.decode(ActionCodec.play(card)).card.id==card
        a=ActionCodec.decode(ActionCodec.play(card,ekka=True));assert a.card.id==card and a.ekka
    assert ActionCodec.decode(ActionCodec.bid('hokum',Suit.CLUBS)).trump_suit==Suit.CLUBS
    assert ActionCodec.decode(ActionCodec.escalation('gahwa')).name=='gahwa'


def _active(rules):
    e=BalootEngine(rules);s=e.create_round(seed=40);e.shuffle_and_initial_deal(s);e.bid_and_complete_deal(s,forced_contract=Contract.HOKUM,forced_buyer=Seat.SOUTH,forced_trump=Suit.HEARTS);e.prepare_projects(s)
    return e,s


def test_observation_shape_and_no_hidden_hand_dependence(rules):
    e,s=_active(rules);info=InformationState.from_complete(s,s.current_player)
    features=encode_information_state(info)
    assert features.shape==(FEATURE_COUNT,) and len(feature_names())==FEATURE_COUNT
    modified=s.clone()
    # Swap two fully hidden opponent cards without altering public knowledge.
    opponents=[seat for seat in Seat if seat!=s.current_player]
    a,b=opponents[:2];ca=next(iter(modified.hand(a)));cb=next(iter(modified.hand(b)))
    modified.hands[int(a)]^=(1<<ca.id)|(1<<cb.id);modified.hands[int(b)]^=(1<<cb.id)|(1<<ca.id)
    info2=InformationState.from_complete(modified,s.current_player)
    assert (encode_information_state(info)==encode_information_state(info2)).all()


def test_environment_completes_round_legally(rules):
    env=BalootEnvironment(rules);result=env.reset(seed=15,contract=Contract.SUN,buyer=Seat.NORTH)
    steps=0
    while not result.terminated:
        action=next(i for i,v in enumerate(result.legal_mask) if v and i<32)
        result=env.step(action);steps+=1
    assert steps==32 and env.state.round_score is not None


def test_native_legal_move_parity_many_states(rules):
    if os.name == "nt":
        pytest.skip("ELF native parity is verified by the WSL release gate on Windows hosts")
    native=NativeAccelerator(rules)
    engine=BalootEngine(rules)
    for contract,trump in ((Contract.SUN,None),(Contract.HOKUM,Suit.HEARTS),(Contract.ASHKAL,None)):
        for seed in range(8):
            s=engine.create_round(seed=seed+100);engine.shuffle_and_initial_deal(s);buyer=Seat.NORTH if contract==Contract.ASHKAL else Seat.SOUTH;engine.bid_and_complete_deal(s,forced_contract=contract,forced_buyer=buyer,forced_trump=trump);engine.prepare_projects(s);bot=HeuristicBot(engine,BotProfile(s.seed))
            while s.current_player is not None:
                assert set(native.legal_moves(s))==set(engine.legal_moves(s))
                engine.play_card(s,bot.choose(s))


def test_selfplay_and_content_addressed_dataset(root, rules, tmp_path):
    from baloot_platform.config import build_bundle
    from baloot_platform.ml.dataset import DatasetWriter
    from baloot_platform.ml.selfplay import play_round
    first=play_round(rules,seed=818,contract=Contract.SUN,buyer=Seat.NORTH,trump_suit=None)
    second=play_round(rules,seed=818,contract=Contract.SUN,buyer=Seat.NORTH,trump_suit=None)
    assert first.steps==32 and first.examples==second.examples
    bundle=build_bundle(root)
    writer=DatasetWriter(tmp_path/'datasets')
    a=writer.write(first.examples,rules_hash=rules.sha256,bundle_hash=bundle.sha256,purpose='selfplay-test')
    b=writer.write(first.examples,rules_hash=rules.sha256,bundle_hash=bundle.sha256,purpose='selfplay-test')
    assert a.sha256==b.sha256 and a.example_count==32 and a.examples_path.read_text().count('\n')==32


def test_uniform_model_masks_illegal_actions(rules):
    from baloot_platform.ml.models import UniformLegalModel
    env=BalootEnvironment(rules);result=env.reset(seed=919,contract=Contract.HOKUM,buyer=Seat.SOUTH,trump_suit=Suit.CLUBS)
    output=UniformLegalModel().infer(result.observation,result.legal_mask)
    assert output.policy_logits.shape==(ACTION_COUNT,)
    assert all(output.policy_logits[i] < -1e8 for i,legal in enumerate(result.legal_mask) if not legal)

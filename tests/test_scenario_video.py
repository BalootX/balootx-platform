from pathlib import Path

import yaml

from baloot_platform.core.enums import Contract, Seat, Suit
from baloot_platform.core.scenario import ScenarioRequest, generate_episode
from baloot_platform.video.remotion_renderer import RemotionProfessionalRenderer


def test_episode_contains_full_lifecycle(rules,convention):
    ep=generate_episode(rules,convention,bundle_hash=rules.sha256,request=ScenarioRequest.governed(seed=1,contract=Contract.HOKUM,buyer=Seat.SOUTH,trump_suit=Suit.HEARTS,require_project=True,requested_samples=4,exact_allocation_limit=200,exact_search_max_remaining=4,max_seed_attempts=20))
    data=ep.to_dict();events={e['type'] for e in data['events']}
    assert len(data['completedTricks'])==8
    assert {'deck_shuffled','purchase_card_exposed','contract_purchased','final_deal_complete','play_started','round_scored'}<=events
    assert data['buyer']=='south' and data['firstLeader']=='north'
    assert 0<ep.pause_before_play_index<32


def test_renderer_produces_observer_safe_governed_request(root,rules,convention):
    ep=generate_episode(rules,convention,bundle_hash=rules.sha256,request=ScenarioRequest.governed(seed=1,contract=Contract.HOKUM,buyer=Seat.SOUTH,trump_suit=Suit.HEARTS,require_project=True,requested_samples=2,exact_allocation_limit=50,exact_search_max_remaining=2,max_seed_attempts=20))
    profile=yaml.safe_load((root/'configs/video/premium.yaml').read_text(encoding='utf-8'))
    renderer=RemotionProfessionalRenderer(profile, repository_root=root)
    render_bundle=renderer.build_render_request(ep, export_preset='hooked_flagship')
    assert render_bundle.verify()
    assert all(scene['observer'] == 'south' for scene in render_bundle.value['scenes'])
    media=render_bundle.value['render']
    assert (media['width'], media['height'], media['fps']) == (1080, 1920, 60)
    assert media['codec'] == 'libx264' and media['pixel_format'] == 'yuv420p'
    assert media['audio_rate'] == 48000
    assert render_bundle.value['timeline']['totalFrames'] == 4320

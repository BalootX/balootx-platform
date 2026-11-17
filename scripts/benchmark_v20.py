#!/usr/bin/env python3
from __future__ import annotations
import json, os, platform, time
from pathlib import Path
import yaml
from fastapi.testclient import TestClient
from baloot_platform.analysis.models import InformationState
from baloot_platform.config import build_bundle
from baloot_platform.core.engine import BalootEngine
from baloot_platform.core.episode_io import load_episode_spec
from baloot_platform.live import LiveRoom, SQLiteLiveStore
from baloot_platform.native import NativeAccelerator
from baloot_platform.performance import PerformanceService
from baloot_platform.service.container import build_container
from baloot_platform.service.mcp_server import MCPProtocol

ROOT=Path(__file__).resolve().parents[1]

def main()->None:
    profile=yaml.safe_load((ROOT/'configs/performance/budgets_v20.yaml').read_text(encoding='utf-8'))
    service=PerformanceService(profile)
    bundle=build_bundle(ROOT)
    episode=load_episode_spec(ROOT/'examples/v7-outcome-flip-source.json',bundle.rules)
    engine=BalootEngine(bundle.rules); state=episode.state_before_pause.clone()
    runtime=ROOT/'.release-runtime/v20-benchmark'; runtime.mkdir(parents=True,exist_ok=True)
    native=NativeAccelerator(bundle.rules, ROOT/'build/native-release/libbaloot_native.so')
    container=build_container(root=ROOT,writable_root=runtime)
    sqlite_live_synchronous=str(profile["measurement"]["sqlite_live_synchronous"]).strip().upper()
    live_store=SQLiteLiveStore(
        runtime/'live-benchmark.sqlite3',
        synchronous=sqlite_live_synchronous,
    )
    protocol=MCPProtocol(); protocol.process({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'benchmark','version':'1'}}})
    try:
        current=container.configuration.store.list_bundles(status='published',limit=1)[0]
        legal=engine.legal_moves(state)
        if not legal: raise RuntimeError('benchmark state has no legal card')
        action_states=[state.clone() for _ in range(220)]
        action_index={'value':0}
        def apply_action():
            clone=action_states[action_index['value']]
            action_index['value'] += 1
            engine.play_card(clone,legal[0])
        def observer_state():
            InformationState.from_complete(state,state.current_player).to_dict()
        def live_cas():
            room=LiveRoom.create(game_id='BENCH-GAME',owner_subject_id='BENCH')
            room.append_event('room_created',subject_id='BENCH'); live_store.save(room)
        operation_registry=yaml.safe_load((ROOT/'configs/capabilities/platform_v20.yaml').read_text(encoding='utf-8'))
        operations={
          'legal_moves_p95_ms':lambda: engine.legal_moves(state),
          'native_legal_moves_p95_ms':lambda: native.legal_moves(state),
          'native_apply_undo_p95_ms':lambda: native.apply_undo_roundtrip(state,legal[0]),
          'apply_action_p95_ms':apply_action,
          'observer_state_p95_ms':observer_state,
          'configuration_resolve_p95_ms':lambda: container.configuration.resolve(revision_id=current.revision_id),
          'plugin_registry_list_p95_ms':lambda: container.registry.list(),
          'operation_registry_p95_ms':lambda: operation_registry['operations']['analysis'],
          'mcp_tool_dispatch_p95_ms':lambda: protocol.process({'jsonrpc':'2.0','id':2,'method':'ping','params':{}}),
          'live_cas_p95_ms':live_cas,
        }
        measurements=[]
        for name,fn in operations.items():
            key=name
            if name=='native_legal_moves_p95_ms': key='legal_moves_p95_ms'
            if name=='plugin_registry_list_p95_ms': key='operation_registry_p95_ms'
            iterations=500 if 'legal_moves' in name or 'apply_undo' in name else 100
            if name=='live_cas_p95_ms': iterations=40
            measurements.append(service.measure(name,fn,iterations=iterations,warmup=10,budget_key=key))
        os.environ.setdefault('BALOOT_ENV','test')
        from baloot_platform.service.api import app
        client=TestClient(app)
        measurements.append(service.measure('api_health_p95_ms',lambda: client.get('/api/v1/health').raise_for_status(),iterations=40,warmup=5))
        report=service.report(measurements)
        by_name={m.name:m for m in measurements}
        report.update({
          'schemaVersion':2,'version':'20.0.0','bundleHash':bundle.sha256,
          'hardware':{'processor':platform.processor(),'machine':platform.machine(),'python':platform.python_version()},
          'storageConfiguration':{
            'backend':'sqlite',
            'journalMode':'WAL',
            'synchronous':sqlite_live_synchronous,
            'durabilityBoundary':(
              'NORMAL preserves database consistency but the latest committed WAL transactions can be lost '
              'after operating-system or power failure; SQLiteLiveStore defaults to FULL'
            ),
          },
          'throughput':{
            'pythonLegalPositionsPerSecond':round(1000.0/max(1e-6,by_name['legal_moves_p95_ms'].median_ms),2),
            'nativeLegalPositionsPerSecond':round(1000.0/max(1e-6,by_name['native_legal_moves_p95_ms'].median_ms),2),
            'nativeApplyUndoPerSecond':round(1000.0/max(1e-6,by_name['native_apply_undo_p95_ms'].median_ms),2),
          },
          'proofBoundary':(
            'local single-process benchmark on the declared execution environment and storage configuration; '
            'not deployment-scale or power-loss durability proof'
          ),
        })
        output=ROOT/'release/performance-benchmark-v20.json'; output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(output)
        if report['status']!='passed': raise SystemExit('performance budgets failed')
    finally:
        container.shutdown()
if __name__=='__main__': main()

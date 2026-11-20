#!/usr/bin/env python3
"""Hermetic verification lanes for Baloot Platform 19.0.0."""
from __future__ import annotations
import hashlib, json, os, re, shutil, socket, subprocess, sys, tempfile, time, tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.release_html import script_blocks
VERSION=str(tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']['version'])
BASENAME='baloot-platform-v19-premium-outcome-flip'
REPORT=ROOT/'release/verification-report-v19.json'
GENERIC=ROOT/'release/verification-report.json'
PROPERTY=ROOT/'release/property-check-v19.json'

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def utc_now()->str:
    epoch=int(os.environ.get('SOURCE_DATE_EPOCH','315532800'))
    return datetime.fromtimestamp(epoch,timezone.utc).isoformat().replace('+00:00','Z')

def run(name:str,command:list[str],*,cwd:Path=ROOT,env:dict[str,str]|None=None,timeout:int=1200)->dict[str,Any]:
    merged=os.environ.copy(); merged.update({'PYTHONPATH':str(cwd/'src'),'BALOOT_ROOT':str(cwd),'SOURCE_DATE_EPOCH':'315532800'})
    if env: merged.update(env)
    start=time.perf_counter()
    try:
        p=subprocess.run(command,cwd=cwd,env=merged,text=True,capture_output=True,timeout=timeout)
        rc=p.returncode; out=p.stdout; err=p.stderr
    except subprocess.TimeoutExpired as exc:
        rc=124; out=exc.stdout.decode() if isinstance(exc.stdout,bytes) else (exc.stdout or ''); err=exc.stderr.decode() if isinstance(exc.stderr,bytes) else (exc.stderr or '')
        err += f'\nTimed out after {timeout}s'
    result={'name':name,'command':command,'status':'passed' if rc==0 else 'failed','returnCode':rc,'durationSeconds':round(time.perf_counter()-start,3),'stdout':out[-20000:],'stderr':err[-20000:]}
    print(f"[{result['status']}] {name} ({result['durationSeconds']}s)",flush=True)
    if rc: raise RuntimeError(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def free_port()->int:
    with socket.socket() as s: s.bind(('127.0.0.1',0)); return int(s.getsockname()[1])

def grpc_probe()->dict[str,Any]:
    port=free_port()
    script=r'''
import json,os,subprocess,sys,tempfile
import grpc
from baloot_platform.generated import platform_pb2,platform_pb2_grpc,shared_pb2,shared_pb2_grpc,operations_pb2,operations_pb2_grpc,hitl_pb2,hitl_pb2_grpc,configuration_pb2,configuration_pb2_grpc
from baloot_platform.version import __version__
root=os.environ['BALOOT_ROOT']; port=os.environ['BALOOT_GRPC_PORT']; runtime=tempfile.mkdtemp(prefix='grpc-v19-')
env=os.environ.copy(); env.update({'BALOOT_GRPC_BIND':'127.0.0.1:'+port,'BALOOT_DATA_DIR':runtime,'BALOOT_RUNTIME':runtime,'BALOOT_ENV':'test'})
proc=subprocess.Popen([sys.executable,'-m','baloot_platform.service.grpc_service'],cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
try:
 channel=grpc.insecure_channel('127.0.0.1:'+port); grpc.channel_ready_future(channel).result(timeout=30)
 platform=platform_pb2_grpc.BalootPlatformStub(channel); health=platform.HealthCheck(platform_pb2.Empty(),timeout=10)
 assert health.status=='ok' and health.version==__version__=='19.0.0'
 shared=shared_pb2_grpc.SharedPlatformStub(channel); listed=shared.ListBoards(shared_pb2.ListBoardsRequest(limit=5),timeout=10)
 payload=json.loads(listed.canonical_json.decode()); assert isinstance(payload.get('items'),list)
 assert payload['terminology']=={'matchAr':'صكّة','roundAr':'لفة','trickAr':'أكلة'}
 operations=operations_pb2_grpc.BalootOperationsStub(channel)
 op_health=operations.HealthCheck(operations_pb2.Empty(),timeout=10); assert op_health.status=='ok' and op_health.version==__version__
 security=operations.SecurityCapabilities(operations_pb2.Empty(),timeout=10)
 security_payload=json.loads(security.canonical_json.decode()); assert 'scopes' in security_payload and 'baloot.live' in security_payload['scopes']
 hitl=hitl_pb2_grpc.HumanLearningStub(channel); hitl_health=hitl.HealthCheck(hitl_pb2.Empty(),timeout=10)
 assert hitl_health.status=='ok' and hitl_health.version==__version__
 hitl_profile=json.loads(hitl.Profile(hitl_pb2.Empty(),timeout=10).canonical_json.decode())
 assert hitl_profile['automaticProductionApply'] is False and hitl_profile['automaticModelPromotion'] is False
 configuration=configuration_pb2_grpc.ConfigurationControlStub(channel)
 capabilities=json.loads(configuration.GetCapabilities(configuration_pb2.Empty(),timeout=10).canonical_json.decode())
 assert capabilities['automaticProductionApply'] is False
 listed_profiles=json.loads(configuration.ListProfiles(configuration_pb2.ProfileQuery(limit=5),timeout=10).canonical_json.decode())
 assert isinstance(listed_profiles.get('items'),list)
 plugins=json.loads(configuration.ListPlugins(configuration_pb2.PluginQuery(),timeout=10).canonical_json.decode())
 assert plugins.get('externalInProcessAllowed') is False
 print(json.dumps({'version':health.version,'boards':len(payload['items']),'securityScopes':len(security_payload['scopes']),'hitl':True,'configurationProfiles':len(listed_profiles['items']),'plugins':len(plugins.get('items',[]))},ensure_ascii=False)); channel.close()
finally:
 proc.terminate()
 try: proc.wait(timeout=10)
 except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=5)
'''
    return run('typed-grpc-v19',[sys.executable,'-c',script],env={'BALOOT_GRPC_PORT':str(port)},timeout=90)

def mcp_probe()->dict[str,Any]:
    script=r'''
import json,os,subprocess,sys,tempfile
from fastapi.testclient import TestClient
from baloot_platform.service.mcp_server import create_http_app
from baloot_platform.version import __version__
root=os.environ['BALOOT_ROOT']; runtime=tempfile.mkdtemp(prefix='mcp-v19-')
env=os.environ.copy(); env.update({'PYTHONPATH':root+'/src','BALOOT_ROOT':root,'BALOOT_ENV':'test','BALOOT_DATA_DIR':runtime,'BALOOT_RUNTIME':runtime})
messages=[{'jsonrpc':'2.0','id':1,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'v19','version':'1'}}},{'jsonrpc':'2.0','method':'notifications/initialized'},{'jsonrpc':'2.0','id':2,'method':'tools/list','params':{}},{'jsonrpc':'2.0','id':3,'method':'resources/list','params':{}}]
p=subprocess.run([sys.executable,'-m','baloot_platform.service.mcp_server'],input=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in messages),text=True,capture_output=True,cwd=root,env=env,timeout=60,check=True)
responses=[json.loads(x) for x in p.stdout.splitlines() if x.strip()]; assert [x['id'] for x in responses]==[1,2,3]
assert responses[0]['result']['serverInfo']['version']==__version__=='19.0.0'
names={x['name'] for x in responses[1]['result']['tools']}; uris={x['uri'] for x in responses[2]['result']['resources']}
required={'list_shared_boards','create_shared_board','verify_shared_board','create_shared_room','start_shared_run','reopen_shared_run','verify_shared_run','play_shared_action','compare_shared_runs','explain_shared_decision','review_shared_decision','list_live_rooms','create_live_room','get_live_room','claim_live_seat','set_live_ready','start_live_table','play_live_action','verify_live_room','list_adjudication_cases','open_adjudication_case','get_adjudication_case','recommend_adjudication_ruling','get_security_capabilities','get_hitl_profile','list_hitl_cases','create_hitl_case','get_hitl_case','assign_hitl_review','submit_hitl_review','compute_hitl_consensus','set_hitl_consent','create_hitl_training_candidate','record_hitl_challenger_evaluation','render_hitl_review_clip'}
required |= {'get_platform_health','list_platform_capabilities','get_configuration_capabilities','list_configuration_profiles','create_configuration_profile','publish_configuration_profile','resolve_configuration_profile','compare_configuration_profiles','list_configuration_bundles','create_configuration_bundle','publish_configuration_bundle','resolve_configuration_bundle','get_performance_profile','run_performance_quick'}
assert required<=names
assert {'baloot://shared/profile','baloot://adjudication/profile','baloot://security/capabilities','baloot://hitl/profile','baloot://configuration/registry','baloot://plugins/policy','baloot://performance/profile'}<=uris
os.environ.update({'BALOOT_ROOT':root,'BALOOT_DATA_DIR':runtime,'BALOOT_RUNTIME':runtime,'BALOOT_ENV':'test','BALOOT_MCP_AUTH_TOKEN':'v19-secret'})
client=TestClient(create_http_app()); req={'jsonrpc':'2.0','id':7,'method':'initialize','params':{'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'http','version':'1'}}}
assert client.post('/mcp',json=req).status_code==401
reply=client.post('/mcp',json=req,headers={'Authorization':'Bearer v19-secret','Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'})
assert reply.status_code==200 and reply.json()['result']['serverInfo']['version']==__version__; assert client.get('/mcp').status_code==405
print(json.dumps({'tools':len(names),'resources':len(uris),'http':reply.status_code},ensure_ascii=False))
'''
    return run('mcp-v19-stdio-http',[sys.executable,'-c',script],timeout=120)

def studio_js_checks()->list[dict[str,Any]]:
    out=[]; directory=ROOT/'release/studio-js-v19'; shutil.rmtree(directory,ignore_errors=True); directory.mkdir(parents=True)
    for html in sorted((ROOT/'src/baloot_platform/service/static').rglob('*.html')):
        blocks=script_blocks(html.read_text(encoding='utf-8'))
        for i,block in enumerate(blocks):
            safe='-'.join(html.relative_to(ROOT/'src/baloot_platform/service/static').with_suffix('').parts); path=directory/f'{safe}-{i}.js'; path.write_text(block,encoding='utf-8')
            out.append(run(f'js-{html.stem}-{i}',['node','--check',str(path)]))
    return out

def docker_static()->dict[str,Any]:
    import yaml
    start=time.perf_counter()
    configured=os.environ.get('BALOOT_DEPLOYMENT_CONTRACT_DIR', '').strip()
    if not configured:
        raise RuntimeError('private deployment contract directory must be supplied explicitly')
    contract_root=Path(configured).resolve(strict=True)
    compose=yaml.safe_load((contract_root/'docker-compose.yml').read_text()); prod=yaml.safe_load((contract_root/'docker-compose.production.yml').read_text()); docker=(ROOT/'Dockerfile').read_text()
    for name in ('api','video-worker','grpc','mcp'):
        item=compose['services'][name]; assert item.get('read_only') is True; assert 'ALL' in item.get('cap_drop',[]); assert 'no-new-privileges:true' in item.get('security_opt',[]); assert item['image'].endswith(':19.0.0')
    assert {'postgres','migrate'}<=set(prod['services']); assert 'USER 10001' in docker; assert 'healthz' in docker
    return {'name':'docker-static-contract','status':'passed','durationSeconds':round(time.perf_counter()-start,3),'daemonAvailable':shutil.which('docker') is not None}

def media_check()->dict[str,Any]:
    path=ROOT/'outputs'/f'{BASENAME}.mp4'
    probe=run('ffprobe-v19',['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_space,color_transfer,color_primaries,sample_rate,channels','-of','json',str(path)])
    data=json.loads(probe['stdout']); video=next(x for x in data['streams'] if x.get('codec_type')=='video'); audio=next(x for x in data['streams'] if x.get('codec_type')=='audio')
    assert video['codec_name']=='h264' and (video['width'],video['height'])==(1080,1920) and video['r_frame_rate']==video['avg_frame_rate']=='60/1' and video['pix_fmt']=='yuv420p'; assert all(video.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries')); assert audio['codec_name']=='aac' and audio['sample_rate']=='48000' and audio['channels']==2
    run('decode-v19',['ffmpeg','-v','error','-i',str(path),'-f','null','-'],timeout=600)
    loudness = run('audio-loudness-v19',[
        'ffmpeg','-nostats','-i',str(path),'-filter_complex','ebur128=peak=true','-f','null','-'
    ],timeout=600)
    summary = loudness['stderr']
    integrated_matches = re.findall(r'\bI:\s*(-?\d+(?:\.\d+)?) LUFS', summary)
    peak_matches = re.findall(r'Peak:\s*(-?\d+(?:\.\d+)?) dBFS', summary)
    if not integrated_matches or not peak_matches:
        raise AssertionError('unable to parse EBU R128 audio summary')
    integrated = float(integrated_matches[-1]); peak = float(peak_matches[-1])
    assert -20.0 <= integrated <= -14.0, integrated
    assert peak <= -0.5, peak
    loudness_payload = {'integratedLufs': integrated, 'truePeakDbfs': peak, 'targetLufs': float(ROOT.joinpath('configs/video/premium_motion_v16.yaml').read_text().split('target_lufs:',1)[1].splitlines()[0].strip())}
    (ROOT/'release/audio-loudness-v19.json').write_text(json.dumps(loudness_payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    
    manifest=json.loads((ROOT/'outputs'/f'{BASENAME}-manifest.json').read_text(encoding='utf-8'))
    motion=manifest.get('motionContract',{})
    assert manifest['renderer']['id']=='premium_motion_v16' and manifest['renderer']['version']=='19.0.0'
    assert motion.get('perFrameInterpolation') is True and motion.get('trickCollection') is True and motion.get('trickWinnerText') is False
    assert motion.get('winnerYellowGlow') is True and int(motion.get('flightSteps',0)) >= 30
    presentation=manifest.get('presentationContract',{})
    assert presentation.get('identityPill') is True and presentation.get('qrSerial') is True
    assert presentation.get('safeOverlayLayout') is True and presentation.get('redundantSeatRelationLabels') is False
    assert presentation.get('winnerText') is False and presentation.get('preflightSourceOnly') is True
    prior_video=os.environ.get('BALOOT_PRIOR_PREMIUM_VIDEO')
    if prior_video and Path(prior_video).is_file(): assert sha256(path) != sha256(Path(prior_video)), 'V19 video unexpectedly matches prior media'
    probe['parsed']=data; probe['loudness']=loudness_payload; probe['motionContract']=motion; probe['presentationContract']=presentation; (ROOT/'release/media-probe-v19.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); return probe

def hitl_media_check()->dict[str,Any]:
    base=ROOT/'outputs'/'baloot-platform-v19-hitl-review-clip'
    video=Path(str(base)+'.mp4'); manifest_path=Path(str(base)+'-manifest.json')
    probe=run('ffprobe-hitl-v19',['ffprobe','-v','error','-show_entries','format=duration,size:stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,sample_rate,channels','-of','json',str(video)])
    data=json.loads(probe['stdout']); stream=next(x for x in data['streams'] if x.get('codec_type')=='video')
    assert stream['codec_name']=='h264' and (stream['width'],stream['height'])==(720,1280) and stream['r_frame_rate']==stream['avg_frame_rate']=='30/1'
    run('decode-hitl-v19',['ffmpeg','-v','error','-i',str(video),'-f','null','-'],timeout=300)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['observerSafe'] is True and manifest['hiddenHandsIncluded'] is False
    assert manifest['productionRecommendationIncluded'] is False and manifest['crowdConsensusIncluded'] is False
    payload={'probe':data,'manifest':manifest}; (ROOT/'release/hitl-review-probe-v19.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return {'name':'hitl-review-media','status':'passed','durationSeconds':0.0,'manifest':manifest}

def clean_wheel(wheel:Path)->dict[str,Any]:
    with tempfile.TemporaryDirectory(prefix='baloot-v19-wheel-') as temp:
        target=Path(temp)/'site'; run('wheel-install',[sys.executable,'-m','pip','install','--no-deps','--target',str(target),str(wheel)],timeout=300)
        script=r'''
from baloot_platform.version import __version__
from baloot_platform.config import build_bundle,platform_root
from baloot_platform.video.card_assets import create_card_art,RasterCardArt
root=platform_root(); bundle=build_bundle(root); art=create_card_art(bundle.video,120,168)
assert __version__=='19.0.0' and bundle.video['renderer_plugin']=='premium_motion_v16' and bundle.video['card_assets']['pack_id']=='goodall_modern_baloot_32' and isinstance(art,RasterCardArt) and len(art.manifest['cards'])==32 and art.face(0).size==(120,168)
print(__version__,bundle.sha256,art.pack_id)
'''
        return run('wheel-runtime',[sys.executable,'-c',script],cwd=Path(temp),env={'PYTHONPATH':str(target),'BALOOT_ROOT':''},timeout=300)

def main()->None:
    checks=[]; release=ROOT/'release'; release.mkdir(exist_ok=True)
    checks.append(run('sync-package-resources',[sys.executable,'scripts/sync_package_resources.py']))
    checks.append(run('profile-manifest',[sys.executable,'-c',"from pathlib import Path; from baloot_platform.config import write_profile_manifest; write_profile_manifest(Path('.').resolve())"]))
    checks.append(run('openapi',[sys.executable,'scripts/generate_openapi.py']))
    checks.append(run('capability-parity-v19',[sys.executable,'scripts/generate_capability_parity_v19.py']))
    checks.append(run('compileall',[sys.executable,'-m','compileall','-q','src','scripts','sdk/python']))
    for kind,flags,san in (('release','',None),('asan','-fsanitize=address -fno-omit-frame-pointer','address'),('ubsan','-fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer','undefined')):
        directory=ROOT/f'build/native-{kind}'; shutil.rmtree(directory,ignore_errors=True)
        cmd=['cmake','-S','native','-B',str(directory),'-DCMAKE_BUILD_TYPE='+('Release' if kind=='release' else 'Debug')]
        if flags: cmd += [f'-DCMAKE_CXX_FLAGS={flags}',f'-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize={san}',f'-DCMAKE_EXE_LINKER_FLAGS=-fsanitize={san}']
        checks.append(run(f'native-{kind}-configure',cmd)); checks.append(run(f'native-{kind}-build',['cmake','--build',str(directory),'-j1']))
        if kind == 'asan':
            asan_runtime = subprocess.run(
                ['gcc', '-print-file-name=libasan.so'],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            preload = asan_runtime
            existing_preload = os.environ.get('LD_PRELOAD', '').strip()
            if existing_preload:
                preload = asan_runtime + ':' + existing_preload
            env = {
                'ASAN_OPTIONS': 'detect_leaks=1:halt_on_error=1',
                'LD_PRELOAD': preload,
            }
        elif kind == 'ubsan':
            env = {'UBSAN_OPTIONS': 'print_stacktrace=1:halt_on_error=1'}
        else:
            env = None
        checks.append(run(f'native-{kind}-test',['ctest','--test-dir',str(directory),'--output-on-failure'],env=env))
    pytest_path=release/'pytest-lanes-v19.json'
    if os.environ.get('BALOOT_REUSE_TESTS')=='1' and pytest_path.exists():
        pytest_report=json.loads(pytest_path.read_text()); assert pytest_report.get('status')=='passed' and int(pytest_report.get('testFiles',0))==len(list((ROOT/'tests').glob('test_*.py')))
        tests_passed=int(pytest_report['testsPassed']); checks.append({'name':'pytest-lanes','status':'passed','reused':True,'testsPassed':tests_passed,'durationSeconds':0.0})
    else:
        checks.append(run('pytest-lanes',[sys.executable,'scripts/run_test_lanes_v19.py'],timeout=3600)); pytest_report=json.loads(pytest_path.read_text()); tests_passed=int(pytest_report['testsPassed'])
    checks.append(run('property-300',[sys.executable,'scripts/property_check.py','--rounds','300','--native-library',str(ROOT/'build/native-release/libbaloot_native.so'),'--output',str(PROPERTY)],timeout=1200)); shutil.copy2(PROPERTY,release/'property-check.json')
    checks.extend(studio_js_checks()); checks.append(run('typescript-sdk',['tsc','-p','sdk/typescript/tsconfig.json','--noEmit'])); checks.append(run('go-sdk',['go','test','./...'],cwd=ROOT/'sdk/go')); checks.append(grpc_probe()); checks.append(mcp_probe()); checks.append(docker_static())
    shutil.rmtree(ROOT/'dist',ignore_errors=True); (ROOT/'dist').mkdir(); checks.append(run('wheel-build',[sys.executable,'-m','pip','wheel','.','--no-deps','--no-build-isolation','-w','dist'],timeout=600)); wheel=next((ROOT/'dist').glob(f'baloot_platform-{VERSION}-*.whl')); checks.append(clean_wheel(wheel))
    checks.append(run('baseline-model-v19',[sys.executable,'scripts/build_baseline_model_v19.py'],timeout=300))
    checks.append(run('selfplay-league-v19',[sys.executable,'scripts/run_selfplay_league_v19.py','--boards','16'],timeout=600))
    checks.append(run('governance-evidence-v19',[sys.executable,'scripts/generate_v19_governance_evidence.py'],timeout=600))
    checks.append(run('platform-evidence-v19',[sys.executable,'scripts/generate_v19_platform_evidence.py'],timeout=600))
    checks.append(hitl_media_check())
    checks.append(run('render-v19',[sys.executable,'scripts/render_v19_release.py'],timeout=1200)); checks.append(media_check()); checks.append(run('supply-chain',[sys.executable,'scripts/generate_supply_chain.py']))
    artifacts={}; paths=[wheel]
    base=ROOT/'outputs'/BASENAME
    for suffix in ('.mp4','.jpg','.json','.srt','-contact.jpg','-manifest.json','-timeline.json','-comparison.json'):
        paths.append(Path(str(base)+suffix))
    hitl_base=ROOT/'outputs'/'baloot-platform-v19-hitl-review-clip'
    for suffix in ('.mp4','.jpg','-timeline.json','-manifest.json'):
        paths.append(Path(str(hitl_base)+suffix))
    paths += [release/'sbom.cdx.json',release/'license-inventory.json',PROPERTY,release/'pytest-lanes-v19.json',release/'media-probe-v19.json',release/'audio-loudness-v19.json',release/'hitl-review-probe-v19.json',release/'rules-certification-candidate-v19.json',release/'intelligence-benchmark-v19.json',release/'slo-evaluation-v19.json',release/'selfplay-league-v19.json',release/'models/v19-baseline/manifest.json',release/'models/v19-baseline/numpy_multitask_linear_v19.npz',release/'platform-v19-evidence.json',release/'configuration-registry-v19.json',release/'plugin-host-v19.json',release/'performance-benchmark-v19.json',ROOT/'generated/capability-parity-v19.json']
    for path in paths:
        if path.exists(): artifacts[str(path.relative_to(ROOT))]={'sha256':sha256(path),'bytes':path.stat().st_size}
    report={'schemaVersion':1,'project':'baloot-platform','version':VERSION,'generatedAt':utc_now(),'status':'passed','testsPassed':tests_passed,'propertyRounds':300,'checks':checks,'artifacts':artifacts,'terminology':{'matchAr':'صكّة','roundAr':'لفة','trickAr':'أكلة'},'externalValidationBoundaries':{'dockerDaemonAvailable':shutil.which('docker') is not None,'postgresServerAvailable':shutil.which('postgres') is not None,'officialRulesCertification':'external referee governance','livePostgresFailover':'requires a real PostgreSQL deployment','independentSecurityReview':'requires an independent assessment','eliteModelStrength':'requires expert-labelled data and external human benchmark','approvedHumanVoice':'offline smooth Arabic fallback is bundled; approved recordings remain replaceable through the voice provider interface'}}
    text=json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n'; REPORT.write_text(text,encoding='utf-8'); GENERIC.write_text(text,encoding='utf-8'); print(REPORT)
if __name__=='__main__': main()

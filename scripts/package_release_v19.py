#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,stat,tempfile,tomllib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERSION=str(tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version'])
PREFIX=f'baloot-platform-{VERSION}'
OUTPUT_PREFIX='baloot-platform-v19-premium-outcome-flip'
MANIFEST_REL=Path('release/source-manifest-v19.json')
EXCLUDED_DIRS={'private-inputs','.local-state','.codex-local','.git','.pytest_cache','.runtime','.release-runtime','build','__pycache__','.mypy_cache','.ruff_cache','node_modules','.venv','venv','baloot_platform.egg-info','studio-js','studio-js-v6','studio-js-v7.1','studio-js-v8','studio-js-v9','studio-js-v19','test-lanes','test-lanes-v7.1','test-lanes-v8','test-lanes-v9','test-lanes-v19','archive'}
EXCLUDED_SUFFIXES={'.pyc','.pyo','.pkl','.sqlite','.sqlite3','.pid','.log','.tmp'}

def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def include(p:Path)->bool:
 if not p.is_file(): return False
 rel=p.relative_to(ROOT)
 if any(x in EXCLUDED_DIRS or x.endswith('-runtime') for x in rel.parts): return False
 if p.suffix in EXCLUDED_SUFFIXES: return False
 if rel==MANIFEST_REL: return False
 if rel.parts and rel.parts[0]=='outputs': return p.name.startswith(OUTPUT_PREFIX) or p.name.startswith('baloot-platform-v19-hitl-review-clip')
 if rel.parts and rel.parts[0]=='dist': return p.name.startswith(f'baloot_platform-{VERSION}-')
 if rel.parts and rel.parts[0]=='release':
  if len(rel.parts)>1 and rel.parts[1]=='models': return True
  allowed={'verification-report-v19.json','verification-report.json','property-check-v19.json','property-check.json','pytest-lanes-v19.json','media-probe-v19.json','audio-loudness-v19.json','cleanroom-report-v19.json','cleanroom-report.json','sbom.cdx.json','license-inventory.json','selfplay-league-v19.json','external-validation-boundaries-v19.json','hitl-review-probe-v19.json','rules-certification-candidate-v19.json','intelligence-benchmark-v19.json','slo-evaluation-v19.json','platform-v19-evidence.json','configuration-registry-v19.json','plugin-host-v19.json','performance-benchmark-v19.json'}
  if p.name not in allowed and not p.name.startswith('source-manifest-v19'): return False
 return True

def main()->None:
 ap=argparse.ArgumentParser(); ap.add_argument('--output',default=str(Path(tempfile.gettempdir())/f'{PREFIX}-software-complete.zip')); args=ap.parse_args(); output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.unlink(missing_ok=True)
 files=sorted(p for p in ROOT.rglob('*') if include(p))
 manifest={'schemaVersion':1,'project':'baloot-platform','version':VERSION,'archivePrefix':PREFIX,'terminology':{'matchAr':'صكّة','roundAr':'لفة','trickAr':'أكلة'},'fileCount':len(files),'files':{str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in files}}
 mp=ROOT/MANIFEST_REL; mp.parent.mkdir(exist_ok=True); mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); files.append(mp); files.sort()
 with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in files:
   info=zipfile.ZipInfo(f'{PREFIX}/{p.relative_to(ROOT)}',date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.create_system=3; info.external_attr=((0o755 if p.stat().st_mode&stat.S_IXUSR else 0o644)&0xFFFF)<<16; z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
 checksum=output.with_suffix(output.suffix+'.sha256'); checksum.write_text(f'{sha256(output)}  {output.name}\n',encoding='utf-8'); print(output); print(checksum)
if __name__=='__main__':main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time,tomllib,zipfile
from datetime import datetime,timezone
from pathlib import Path
OUTPUT_PREFIX='baloot-platform-v19-premium-outcome-flip'; MANIFEST=Path('release/source-manifest-v19.json')
def sha256(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(name,cmd,*,cwd,timeout=7200):
 env=os.environ.copy(); env.update({'PYTHONPATH':str(cwd/'src'),'BALOOT_ROOT':str(cwd),'SOURCE_DATE_EPOCH':'315532800'}); start=time.perf_counter(); p=subprocess.run(cmd,cwd=cwd,env=env,text=True,capture_output=True,timeout=timeout); r={'name':name,'status':'passed' if p.returncode==0 else 'failed','returnCode':p.returncode,'durationSeconds':round(time.perf_counter()-start,3),'stdout':p.stdout[-20000:],'stderr':p.stderr[-20000:]};
 if p.returncode: raise RuntimeError(json.dumps(r,ensure_ascii=False,indent=2)); return r
 return r
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('archive'); ap.add_argument('--output',required=True); ap.add_argument('--keep',action='store_true'); args=ap.parse_args(); archive=Path(args.archive).resolve(); report_path=Path(args.output).resolve(); parent=Path(tempfile.mkdtemp(prefix='baloot-v19-cleanroom-',dir=tempfile.gettempdir())); preserved=None; checks=[]
 try:
  with zipfile.ZipFile(archive) as z: assert z.testzip() is None; z.extractall(parent)
  roots=[p for p in parent.iterdir() if p.is_dir()]; assert len(roots)==1; root=roots[0]
  version=str(tomllib.loads((root/'pyproject.toml').read_text())['project']['version']); assert version=='19.0.0'
  manifest=json.loads((root/MANIFEST).read_text()); assert manifest['version']==version
  for rel,meta in manifest['files'].items():
   p=root/rel; assert p.is_file() and p.stat().st_size==meta['bytes'] and sha256(p)==meta['sha256'], rel
  checks.append({'name':'source-manifest','status':'passed','files':len(manifest['files'])})
  before={p.name:sha256(p) for p in (root/'outputs').iterdir() if p.is_file() and (p.name.startswith(OUTPUT_PREFIX) or p.name.startswith('baloot-platform-v19-hitl-review-clip'))}; before_wheels={p.name:sha256(p) for p in (root/'dist').glob('baloot_platform-19.0.0-*.whl')}
  checks.append(run('cleanroom-verification',[sys.executable,'scripts/verify_release_v19.py'],cwd=root,timeout=7200))
  after={p.name:sha256(p) for p in (root/'outputs').iterdir() if p.is_file() and (p.name.startswith(OUTPUT_PREFIX) or p.name.startswith('baloot-platform-v19-hitl-review-clip'))}; assert before==after, {k:(before.get(k),after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)}; checks.append({'name':'byte-identical-render-artifacts','status':'passed','files':len(after),'sha256':after})
  after_wheels={p.name:sha256(p) for p in (root/'dist').glob('baloot_platform-19.0.0-*.whl')}; assert not before_wheels or before_wheels==after_wheels; checks.append({'name':'deterministic-wheel','status':'passed','sha256':after_wheels})
  generated=datetime.fromtimestamp(315532800,timezone.utc).isoformat().replace('+00:00','Z'); report={'schemaVersion':1,'project':'baloot-platform','version':version,'archive':archive.name,'archiveSha256':sha256(archive),'generatedAt':generated,'status':'passed','checks':checks,'terminology':{'matchAr':'صكّة','roundAr':'لفة','trickAr':'أكلة'},'cleanroomRoot':str(root) if args.keep else None}; report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(report_path)
  if args.keep: preserved=root; print(root)
 finally:
  if preserved is None: shutil.rmtree(parent,ignore_errors=True)
if __name__=='__main__':main()

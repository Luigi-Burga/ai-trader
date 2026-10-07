#!/usr/bin/env python3
"""Prediction Tracker V1.6.1 A/B Benchmark R2 - forensic field diff."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
VERSION="1.1-R2.1"; TRACKER_V16_1="1.6.1"; PRODUCTION_MODULE="app/prediction_tracker.py"; TARGET_DATE="2026-10-06"
def sha256_file(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def load_jsonl(p):
 rows=[]
 with p.open(encoding='utf-8') as f:
  for n,line in enumerate(f,1):
   if line.strip(): rows.append(json.loads(line))
 return rows
def copy_tree(s,d): shutil.copytree(s,d)
def age_cache_files(d,sec):
 now=time.time(); n=0
 for p in d.rglob('*'):
  if p.is_file(): os.utime(p,(now-sec,now-sec)); n+=1
 return n
def canonicalize(v):
 if isinstance(v,dict): return {k:canonicalize(x) for k,x in v.items() if k!='evaluated_at_utc'}
 if isinstance(v,list): return [canonicalize(x) for x in v]
 return v
def record_key(r,i):
 for k in ('prediction_id','id','snapshot_id'):
  if r.get(k) is not None: return str(r[k])
 t=r.get('ticker','')
 for k in ('timestamp','as_of','created_at','evaluated_at_utc'):
  if r.get(k): return f'{t}|{r[k]}'
 return f'__index__{i}'
def index_rows(rows):
 out={}
 for i,r in enumerate(rows):
  k=record_key(r,i)
  if k in out: raise RuntimeError(f'Duplicate logical record key: {k}')
  out[k]=r
 return out
def date_matches(r,date):
 return any(date in str(r.get(k)) for k in ('prediction_date_utc','timestamp','as_of','created_at','evaluated_at_utc') if r.get(k) is not None)
def run_tracker(root,module,pred,cache):
 env=os.environ.copy(); env.update({'AI_TRADER_PREDICTIONS_DIR':str(pred),'AI_TRADER_MARKET_CACHE_DIR':str(cache),'AI_TRADER_AUTONOMOUS_PAPER_EXECUTION':'false','PYTHONUNBUFFERED':'1'})
 cmd=[sys.executable,'-m',module[:-3].replace('/','.') ] if module.startswith('app/') else [sys.executable,str(root/module),'--directory',str(pred)]
 st=time.perf_counter(); p=subprocess.run(cmd,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT); return time.perf_counter()-st,p.stdout,p.returncode
def flatten_diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   if k=='evaluated_at_utc': continue
   q=f'{path}.{k}' if path else k
   if k not in a: out.append({'path':q,'kind':'MISSING_IN_PRODUCTION','production':None,'v1_6_1':b[k]})
   elif k not in b: out.append({'path':q,'kind':'MISSING_IN_V1_6_1','production':a[k],'v1_6_1':None})
   else: out.extend(flatten_diff(a[k],b[k],q))
  return out
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b): return [{'path':path,'kind':'LIST_LENGTH_DIFFERENCE','production':a,'v1_6_1':b}]
  out=[]
  for i,(x,y) in enumerate(zip(a,b)): out.extend(flatten_diff(x,y,f'{path}[{i}]'))
  return out
 return [] if a==b else [{'path':path or '<root>','kind':'VALUE_DIFFERENCE','production':a,'v1_6_1':b}]
def collect(d):
 r={}
 for p in sorted(d.glob('predictions_*.jsonl')): r[p.name]=load_jsonl(p)
 if not r: raise RuntimeError(f'No predictions_*.jsonl found in {d}')
 return r
def compare(A,B):
 R={'semantic_output_match':True,'files_compared':len(set(A)|set(B)),'records_production':0,'records_v1_6_1':0,'identical_records':0,'different_records':0,'missing_in_v1_6_1':0,'missing_in_production':0,'field_difference_counts':Counter(),'difference_categories':Counter(),'differences':[],'target_date':{'date':TARGET_DATE,'production_records':0,'v1_6_1_records':0,'different_records':0,'field_difference_counts':Counter()}}
 for name in sorted(set(A)|set(B)):
  ai=index_rows(A.get(name,[])); bi=index_rows(B.get(name,[])); R['records_production']+=len(ai); R['records_v1_6_1']+=len(bi)
  for k in sorted(set(ai)|set(bi)):
   if k not in ai: R['missing_in_production']+=1; R['semantic_output_match']=False; continue
   if k not in bi: R['missing_in_v1_6_1']+=1; R['semantic_output_match']=False; continue
   if canonicalize(ai[k])==canonicalize(bi[k]): R['identical_records']+=1; continue
   ds=flatten_diff(ai[k],bi[k]); R['different_records']+=1; R['semantic_output_match']=False; target=date_matches(ai[k],TARGET_DATE)
   if target: R['target_date']['different_records']+=1
   R['differences'].append({'file':name,'record_key':k,'ticker':ai[k].get('ticker'),'classification':'TARGET_DATE' if target else 'OTHER_DATE','fields':ds})
   for d in ds:
    R['field_difference_counts'][d['path']]+=1; R['difference_categories'][d['kind']]+=1
    if target: R['target_date']['field_difference_counts'][d['path']]+=1
 for files,key in ((A,'production_records'),(B,'v1_6_1_records')): R['target_date'][key]=sum(date_matches(r,TARGET_DATE) for rows in files.values() for r in rows)
 R['field_difference_counts']=dict(sorted(R['field_difference_counts'].items())); R['difference_categories']=dict(sorted(R['difference_categories'].items())); R['target_date']['field_difference_counts']=dict(sorted(R['target_date']['field_difference_counts'].items()))
 return R
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--age-cache-seconds',type=int,default=1200); ap.add_argument('--keep-workdir',action='store_true'); a=ap.parse_args(); root=Path(a.root).resolve()
 prod=root/PRODUCTION_MODULE; cand=root/'prediction_tracker_v1_6_1.py'; sp=root/'data/predictions'; sc=root/'data/market_cache'
 for p in (prod,cand,sp,sc):
  if not p.exists(): raise FileNotFoundError(p)
 src=collect(sp); total=sum(map(len,src.values())); target=sum(date_matches(r,TARGET_DATE) for rows in src.values() for r in rows)
 if not target: raise RuntimeError(f'R2 target-date guard failed: no records found for {TARGET_DATE}')
 print('='*78); print('Prediction Tracker V1.6.1 — A/B Functional Equivalence Benchmark R2'); print(f'Benchmark version: {VERSION}'); print('Production: app/prediction_tracker.py'); print('Candidate : prediction_tracker_v1_6_1.py'); print('Research-only | production unchanged | orders blocked'); print('R2 forensic mode: exact field/value differences'); print('='*78); print(f'Prediction files: {len(src)}'); print(f'Original records: {total}'); print(f'R2 target-date source records ({TARGET_DATE}): {target}')
 wd=Path(tempfile.mkdtemp(prefix='pt_v161_ab_r2_')); print(f'Isolated workdir: {wd}')
 try:
  apd,bpd,ac,bc=[wd/x for x in ('A_predictions','B_predictions','A_cache','B_cache')]; copy_tree(sp,apd); copy_tree(sp,bpd); copy_tree(sc,ac); copy_tree(sc,bc); print(f'Aged isolated cache files: A={age_cache_files(ac,a.age_cache_seconds)} B={age_cache_files(bc,a.age_cache_seconds)}')
  print('\n--- A: Production V1.5 ---'); ta,oa,ra=run_tracker(root,PRODUCTION_MODULE,apd,ac); print(oa); print(f'A exit_code={ra} elapsed={ta:.3f}s');
  if ra: raise RuntimeError('Production V1.5 benchmark failed')
  print('\n--- B: V1.6.1 ---'); tb,ob,rb=run_tracker(root,'prediction_tracker_v1_6_1.py',bpd,bc); print(ob); print(f'B exit_code={rb} elapsed={tb:.3f}s');
  if rb: raise RuntimeError('V1.6.1 benchmark failed')
  A=collect(apd); B=collect(bpd); ca=sum(map(len,A.values())); cb=sum(map(len,B.values()));
  if not(total==ca==cb): raise RuntimeError(f'R2 dataset integrity failed: source={total} A={ca} B={cb}')
  print(f'R2 dataset integrity: source={total} A={ca} B={cb}'); R=compare(A,B); speed=ta/tb if tb else None; imp=(ta-tb)/ta*100 if ta else None
  R.update({'benchmark_version':VERSION,'candidate_version':TRACKER_V16_1,'production_module':PRODUCTION_MODULE,'candidate_module':'prediction_tracker_v1_6_1.py','source_sha256':{'production':sha256_file(prod),'candidate':sha256_file(cand)},'timing_seconds':{'production_v1_5':ta,'v1_6_1':tb},'performance':{'improvement_percent':imp,'speedup':speed},'age_cache_seconds':a.age_cache_seconds,'dynamic_fields_ignored':['evaluated_at_utc'],'orders_or_alpaca':False,'production_source_modified':False})
  out=root/'prediction_tracker_v1_6_1_ab_results_R2.json'; out.write_text(json.dumps(R,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
  print('\n'+'='*78); print('FINAL A/B R2 RESULT'); print('='*78); print(f'Production V1.5 time : {ta:.3f}s'); print(f'V1.6.1 time          : {tb:.3f}s'); print(f'Improvement           : {imp:.2f}%'); print(f'Speedup               : {speed:.2f}x'); print(f'Semantic output match : {R["semantic_output_match"]}'); print(f'Files compared       : {R["files_compared"]}'); print(f'Records A/B           : {R["records_production"]} / {R["records_v1_6_1"]}'); print(f'Identical records     : {R["identical_records"]}'); print(f'Different records     : {R["different_records"]}'); print('\nField difference counts:');
  for p,n in R['field_difference_counts'].items(): print(f'  {p}: {n}')
  td=R['target_date']; print(f'\nTARGET DATE {TARGET_DATE}'); print(f'  Source records     : {target}'); print(f'  Production records : {td["production_records"]}'); print(f'  V1.6.1 records     : {td["v1_6_1_records"]}'); print(f'  Different records  : {td["different_records"]}'); print('  Field differences:');
  for p,n in td['field_difference_counts'].items(): print(f'    {p}: {n}')
  print(f'\nForensic results JSON: {out}'); print('\nFUNCTIONAL EQUIVALENCE: '+('PASS' if R['semantic_output_match'] else 'FAIL')); return 0
 finally:
  if a.keep_workdir: print(f'Keeping isolated workdir: {wd}')
  else: shutil.rmtree(wd,ignore_errors=True)
if __name__=='__main__': raise SystemExit(main())

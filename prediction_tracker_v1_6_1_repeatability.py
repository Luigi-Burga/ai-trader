#!/usr/bin/env python3
"""Prediction Tracker V1.6.1 repeatability test.

Research-only: runs the candidate twice against identical isolated inputs and
compares the resulting prediction datasets. Production files are never modified.
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

TARGET_DATE = "2026-10-06"
CANDIDATE = "prediction_tracker_v1_6_1.py"

def load_jsonl(p: Path):
    rows=[]
    with p.open(encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            if line.strip():
                rows.append(json.loads(line))
    return rows

def collect(d: Path):
    out={}
    for p in sorted(d.glob("predictions_*.jsonl")):
        out[p.name]=load_jsonl(p)
    if not out:
        raise RuntimeError(f"No predictions_*.jsonl found in {d}")
    return out

def canonicalize(v):
    if isinstance(v, dict):
        return {k: canonicalize(x) for k,x in v.items() if k != "evaluated_at_utc"}
    if isinstance(v, list):
        return [canonicalize(x) for x in v]
    return v

def date_matches(r, date):
    return any(date in str(r.get(k)) for k in (
        "prediction_date_utc", "timestamp", "as_of", "created_at", "evaluated_at_utc"
    ) if r.get(k) is not None)

def run_candidate(root, pred, cache):
    env=os.environ.copy()
    env.update({
        "AI_TRADER_PREDICTIONS_DIR": str(pred),
        "AI_TRADER_MARKET_CACHE_DIR": str(cache),
        "AI_TRADER_AUTONOMOUS_PAPER_EXECUTION": "false",
        "PYTHONUNBUFFERED": "1",
    })
    cmd=[sys.executable, str(root / CANDIDATE), "--directory", str(pred)]
    started=time.perf_counter()
    p=subprocess.run(cmd,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    return time.perf_counter()-started,p.stdout,p.returncode

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--age-cache-seconds",type=int,default=1200)
    ap.add_argument("--keep-workdir",action="store_true")
    args=ap.parse_args()
    root=Path(args.root).resolve()
    source_pred=root/"data/predictions"
    source_cache=root/"data/market_cache"
    candidate=root/CANDIDATE
    if not source_pred.exists(): raise FileNotFoundError(source_pred)
    if not source_cache.exists(): raise FileNotFoundError(source_cache)
    if not candidate.exists(): raise FileNotFoundError(candidate)

    source=collect(source_pred)
    source_total=sum(map(len,source.values()))
    target_total=sum(date_matches(r,TARGET_DATE) for rows in source.values() for r in rows)
    if target_total == 0:
        raise RuntimeError(f"Repeatability target-date guard failed: no records found for {TARGET_DATE}")

    print("="*78)
    print("Prediction Tracker V1.6.1 — Repeatability Test")
    print("="*78)
    print(f"Candidate          : {candidate}")
    print(f"Prediction files   : {len(source)}")
    print(f"Source records     : {source_total}")
    print(f"Target-date records: {target_total} ({TARGET_DATE})")
    print("Production modified: NO")
    print("Orders/Alpaca      : BLOCKED")

    wd=Path(tempfile.mkdtemp(prefix="pt_v161_repeat_"))
    print(f"Isolated workdir   : {wd}")
    try:
        p1=wd/"run1_predictions"; p2=wd/"run2_predictions"
        c1=wd/"run1_cache"; c2=wd/"run2_cache"
        shutil.copytree(source_pred,p1); shutil.copytree(source_pred,p2)
        shutil.copytree(source_cache,c1); shutil.copytree(source_cache,c2)
        old=time.time()-args.age_cache_seconds
        for cache in (c1,c2):
            for f in cache.rglob('*'):
                if f.is_file(): os.utime(f,(old,old))

        print("\n--- Run 1: V1.6.1 ---")
        t1,o1,r1=run_candidate(root,p1,c1)
        print(o1)
        print(f"Run 1 exit_code={r1} elapsed={t1:.3f}s")
        if r1: raise RuntimeError("Repeatability run 1 failed")

        print("\n--- Run 2: V1.6.1 ---")
        t2,o2,r2=run_candidate(root,p2,c2)
        print(o2)
        print(f"Run 2 exit_code={r2} elapsed={t2:.3f}s")
        if r2: raise RuntimeError("Repeatability run 2 failed")

        A=collect(p1); B=collect(p2)
        total_a=sum(map(len,A.values())); total_b=sum(map(len,B.values()))
        if total_a != source_total or total_b != source_total:
            raise RuntimeError(f"Dataset integrity failed: source={source_total} run1={total_a} run2={total_b}")

        differences=[]
        for name in sorted(set(A)|set(B)):
            if name not in A or name not in B:
                differences.append((name,"<file>","file_presence",bool(name in A),bool(name in B)))
                continue
            if len(A[name]) != len(B[name]):
                differences.append((name,"<records>","record_count",len(A[name]),len(B[name])))
                continue
            for i,(a,b) in enumerate(zip(A[name],B[name])):
                ca,cb=canonicalize(a),canonicalize(b)
                if ca != cb:
                    differences.append((name,f"record[{i}]","value",ca,cb))

        td_a=sum(date_matches(r,TARGET_DATE) for rows in A.values() for r in rows)
        td_b=sum(date_matches(r,TARGET_DATE) for rows in B.values() for r in rows)
        print("\n"+"="*78)
        print("FINAL REPEATABILITY RESULT")
        print("="*78)
        print(f"Run 1 time         : {t1:.3f}s")
        print(f"Run 2 time         : {t2:.3f}s")
        print(f"Records Run 1/Run 2: {total_a} / {total_b}")
        print(f"Target date A/B    : {td_a} / {td_b}")
        print(f"Differences         : {len(differences)}")
        print("Dynamic ignored     : evaluated_at_utc")
        print("REPEATABILITY       : " + ("PASS" if not differences else "FAIL"))
        if differences:
            print("\nFirst differences:")
            for d in differences[:20]: print(d)
        return 0 if not differences else 2
    finally:
        if args.keep_workdir:
            print(f"Keeping isolated workdir: {wd}")
        else:
            shutil.rmtree(wd,ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())

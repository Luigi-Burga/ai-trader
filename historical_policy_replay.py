#!/usr/bin/env python3
"""Historical, hypothetical policy replay. Never connects to execution or submits orders."""
import argparse, json
from pathlib import Path
from decision_policy_v1 import DecisionInput, evaluate_decision

def num(x):
    try: return float(x) if x is not None else None
    except (TypeError, ValueError): return None

def load_rows(paths):
    rows=[]
    for path in paths:
        p=Path(path)
        with p.open(encoding='utf-8-sig') as f:
            for line_no,line in enumerate(f,1):
                try: r=json.loads(line)
                except Exception: continue
                gate=r.get('decision_gate') or {}
                m=gate.get('metrics') or {}
                if not isinstance(m,dict): m={}
                score=num(r.get('score'))
                if score is None: continue
                rows.append((score,r,m,str(p),line_no))
    return rows

def main():
    ap=argparse.ArgumentParser(description='Historical hypothetical policy test; no orders')
    ap.add_argument('files',nargs='*',help='Optional prediction JSONL files; bundled historical files are used by default')
    ap.add_argument('--output',default='historical_policy_replay_result.json',help='Output JSON path (default: historical_policy_replay_result.json)')
    ap.add_argument('--ticker',default='PLTR',help='Ticker filter (default: PLTR)')
    args=ap.parse_args()
    default_data = Path(__file__).resolve().parent / 'data'
    input_files = args.files or [str(default_data / 'predictions_2026-10-26.jsonl'), str(default_data / 'predictions_2026-10-27.jsonl')]
    missing_files = [p for p in input_files if not Path(p).is_file()]
    if missing_files: raise SystemExit('ERROR: missing input file(s): ' + ', '.join(missing_files))
    rows=load_rows(input_files)
    if args.ticker: rows=[x for x in rows if str(x[1].get('ticker','')).upper()==args.ticker.upper()]
    if not rows: raise SystemExit('ERROR: no usable prediction records found')
    score,r,m,source,line=max(rows,key=lambda x:x[0])
    confidence=num(m.get('trading_confidence',r.get('confidence')))
    if confidence is not None and confidence > 1: confidence/=100.0
    d=DecisionInput(
      symbol=str(r.get('ticker','UNKNOWN')),
      analytical_signal='BUY', # intentionally hypothetical to exercise BUY gates
      score=score,
      probability_20d=num(m.get('historical_probability')),
      reward_risk=num(m.get('reward_risk')),
      confidence=confidence,
      fallback_ratio=num(m.get('fallback_ratio')),
      match_quality=str(m.get('match_quality')).upper() if m.get('match_quality') is not None else None,
      regime=str(m.get('regime',r.get('regime'))).upper() if m.get('regime',r.get('regime')) is not None else None,
      appt=num(m.get('appt_x')),
      benchmark_trust=str(m.get('benchmark_trust',r.get('benchmark_trust'))).upper() if m.get('benchmark_trust',r.get('benchmark_trust')) is not None else None,
      benchmark_decision_grade=m.get('benchmark_grade',r.get('benchmark_trust_decision_grade')),
      data_valid=True,
      entry_valid=bool(num(m.get('entry_price', (r.get('levels') or {}).get('entry_price')))),
      stop_valid=bool(num(m.get('stop', (r.get('levels') or {}).get('stop')))),
      source='HISTORICAL_HYPOTHETICAL_REPLAY')
    out=evaluate_decision(d).to_dict()
    report={
      'test_mode':'HISTORICAL_HYPOTHETICAL_ONLY',
      'orders_submitted':False,
      'source_file':source,
      'source_line':line,
      'prediction_id':r.get('prediction_id'),
      'timestamp_utc':r.get('prediction_timestamp_utc'),
      'ticker':r.get('ticker'),
      'historical_final_signal':r.get('final_signal'),
      'note':'The historical signal is not changed. analytical_signal=BUY is a test-only override to exercise policy gates; this is not a real BUY signal and cannot authorize an order.',
      'metrics_used':{
       'score':score,'probability_20d':d.probability_20d,'reward_risk':d.reward_risk,
       'confidence_fraction':d.confidence,'fallback_ratio':d.fallback_ratio,
       'match_quality':d.match_quality,'regime':d.regime,'appt':d.appt,
       'benchmark_trust':d.benchmark_trust,'benchmark_decision_grade':d.benchmark_decision_grade,
       'entry_valid':d.entry_valid,'stop_valid':d.stop_valid},
      'policy_result':out}
    rendered=json.dumps(report,indent=2,ensure_ascii=False)
    Path(args.output).write_text(rendered + '\n',encoding='utf-8')
    print(rendered)
    print('\nSaved report: ' + str(Path(args.output).resolve()))
if __name__=='__main__': main()

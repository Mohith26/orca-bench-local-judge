import json, collections
rows=[json.loads(l) for l in open('results.jsonl')]
out=[]
out.append("| config | condition | n | API/parse errors | contract violations | median input tokens | unpatched rca_accuracy wrong | credited a root cause (any flag_match) |")
out.append("|---|---|---|---|---|---|---|---|")
for cfg in ("ollama_default","ctx32k"):
  for cond in ("oracle","symptom_only","wrong_cause"):
    rs=[r for r in rows if r['config']==cfg and r['condition']==cond]
    err=sum('error' in r for r in rs)
    viol=sum(r.get('contract_ok') is False for r in rs)
    toks=sorted(r['input_tokens'] for r in rs if r.get('input_tokens'))
    med=toks[len(toks)//2] if toks else None
    wrong=sum(r.get('unpatched_rca_correct') is False for r in rs)
    cred=sum(any(r.get('flag_matches') or []) for r in rs)
    out.append(f"| {cfg} | {cond} | {len(rs)} | {err} | {viol} | {med} | {wrong} | {cred} |")
for cfg in ("ollama_default","ctx32k"):
  rs=[r for r in rows if r['config']==cfg and 'error' not in r]
  bad_scored=[r for r in rs if r.get('contract_ok') is False]
  print(cfg, 'total',len([r for r in rows if r['config']==cfg]),'parsed',len(rs),'violations',len(bad_scored),'unpatched wrong among violations',sum(r.get('unpatched_rca_correct') is False for r in bad_scored))
  # among contract-ok
  ok=[r for r in rs if r.get('contract_ok')]
  for cond in ("oracle","symptom_only","wrong_cause"):
    o=[r for r in ok if r['condition']==cond]
    print('  ok',cond,len(o),'rca correct',sum(r['unpatched_rca_correct'] for r in o))
print("\n".join(out))
for r in rows:
  if 'error' in r: print('ERR',r['config'],r['task'],r['condition'],r['error'][:150])

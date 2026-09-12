"""Extract detailed failure analysis from eval report for interview prep."""
import json
from collections import Counter
from pathlib import Path

report = Path("logs/eval_comprehensive/report_rejudged.json")
data = json.loads(report.read_text(encoding="utf-8"))
results = data["results"]

# --- retrieval_quality failures ---
rqf = []
for r in results:
    d = r.get("dimension_results", {}).get("retrieval_quality", {})
    if d.get("passed") == False and not d.get("skipped"):
        rqf.append(r)
print(f"retrieval_quality: {len(rqf)} failures")

# --- self_eval failures ---
sef = []
ftypes = Counter()
for r in results:
    d = r.get("dimension_results", {}).get("self_eval_accuracy", {})
    if d.get("passed") == False and not d.get("skipped"):
        sef.append(r)
        ftypes[d.get("failure_type", "?")] += 1
print(f"self_eval_accuracy: {len(sef)} failures")
print(f"  types: {dict(ftypes)}")

# --- answer_quality failures ---
aqf = []
for r in results:
    d = r.get("dimension_results", {}).get("answer_quality", {})
    if d.get("passed") == False and not d.get("skipped"):
        aqf.append(r)
print(f"answer_quality: {len(aqf)} failures")

# Show examples
print("\n--- retrieval_quality 样本 ---")
for r in rqf[:3]:
    d = r["dimension_results"]["retrieval_quality"]
    print(f"  [{r['id']}] {r['query'][:40]}")
    print(f"    title: {d.get('title_hits',{})}")
    print(f"    kw:    {d.get('keyword_hits',{})}")

print("\n--- self_eval 样本 ---")
for r in sef[:5]:
    d = r["dimension_results"]["self_eval_accuracy"]
    print(f"  [{r['id']}] {r['query'][:40]}")
    print(f"    expected={d.get('expected')} actual={d.get('actual')} type={d.get('failure_type')}")

print("\n--- answer_quality 样本 ---")
for r in aqf[:5]:
    d = r["dimension_results"]["answer_quality"]
    must = {k:v for k,v in d.get("must_include",{}).items() if v!="found"}
    shd = {k:v for k,v in d.get("should_not_include",{}).items() if v=="found_violation"}
    print(f"  [{r['id']}] {r['query'][:40]}")
    if must: print(f"    缺少: {must}")
    if shd:  print(f"    违规: {shd}")

# Category breakdown
cats = Counter(r.get("category","?") for r in results if r.get("status")=="failed")
print(f"\n--- 按类别统计失败 ---")
for cat, cnt in sorted(cats.items(), key=lambda x:-x[1]):
    total = sum(1 for r in results if r.get("category")==cat)
    print(f"  {cat}: {cnt}/{total}")

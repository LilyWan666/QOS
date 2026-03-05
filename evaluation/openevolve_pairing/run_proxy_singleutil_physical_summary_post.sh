#!/usr/bin/env bash
set -euo pipefail
ROOT="/work/nvme/betu/lily/QOS/evaluation/openevolve_pairing"
PY="/work/nvme/becn/lily/miniconda3/envs/qos_fig11/bin/python"
GROUP_TAG="$1"
UTILS=(30 37 45 52 60 67 74 81 88)
EVAL_UTILS=(30 60 88)
PROXIES=(cnot_ratio measure_diff)

for proxy in "${PROXIES[@]}"; do
  safe_proxy="${proxy}"
  OUT="${ROOT}/openevolve_output/_proxy_${safe_proxy}_singleutil_physical_306088_8196_${GROUP_TAG}"
  mkdir -p "${OUT}"
  declare -A CLEAN_CSVS=()
  for eu in "${EVAL_UTILS[@]}"; do
    raw_csv="${ROOT}/pairing_metadata/pair_metrics_util${eu}_shots8196.csv"
    clean_csv="${OUT}/_clean_pair_metrics_util${eu}_shots8196.csv"
    "${PY}" - "${raw_csv}" "${clean_csv}" <<'PY'
import csv, sys
raw_csv, clean_csv = sys.argv[1], sys.argv[2]
with open(raw_csv, "r", encoding="utf-8", newline="") as fin:
    reader = csv.DictReader(fin)
    if not reader.fieldnames:
        raise RuntimeError(f"Missing header in {raw_csv}")
    fieldnames = [h for h in reader.fieldnames if h is not None]
    with open(clean_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            row.pop(None, None)
            writer.writerow({k: row.get(k, "") for k in fieldnames})
PY
    CLEAN_CSVS["${eu}"]="${clean_csv}"
  done

  methods_tsv="${OUT}/methods.tsv"
  : > "${methods_tsv}"

  for u in "${UTILS[@]}"; do
    patt="${ROOT}/openevolve_output/Qwen2.5-14B-Instruct_iter200_*_u${u}_proxy_${proxy}_iter200_${GROUP_TAG}/best/best_program.py"
    prog=$(ls -1 ${patt} 2>/dev/null | sort | tail -n1 || true)
    if [[ -z "${prog}" ]]; then
      echo "[ERR] missing best_program for proxy=${proxy} util=${u} pattern=${patt}" >&2
      exit 2
    fi
    printf "trainu%s\t%s\n" "${u}" "${prog}" >> "${methods_tsv}"
  done

  # Evaluate on physical util30/60/88
  while IFS=$'\t' read -r method prog; do
    [[ -z "${method}" ]] && continue
    tu="${method#trainu}"
    for eu in "${EVAL_UTILS[@]}"; do
      csv="${CLEAN_CSVS[${eu}]}"
      for tk in 10 20; do
        tag="proxy_${proxy}_trainu${tu}_eval${eu}_t${tk}"
        "${PY}" "${ROOT}/plot_qos_vs_evolved_from_csv.py" \
          --csv "${csv}" --util "${eu}" --shots 8196 --top-k-ratio "${tk}" \
          --orig-target "${ROOT}/target_orig.py" --new-target "${prog}" \
          --out "${OUT}/${tag}.png" \
          --out-csv "${OUT}/${tag}.csv" \
          --bar-out "${OUT}/${tag}_bar.png" \
          --bar-out-csv "${OUT}/${tag}_bar.csv" \
          --summary-csv "${OUT}/${tag}_summary.csv" >/tmp/${tag}.log 2>&1
      done
    done
  done < "${methods_tsv}"

  # Aggregate and plot like depthratio summary
  "${PY}" - << PY
import os, csv, math, numpy as np
import matplotlib.pyplot as plt
out='${OUT}'
proxy='${proxy}'
train_utils=[30,37,45,52,60,67,74,81,88]
eval_utils=[30,60,88]

def wmean(path,mk,ck):
    vals=[]; cnt=[]
    with open(path,'r',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            try:
                v=float(r.get(mk,'nan')); c=float(r.get(ck,'0'))
            except: continue
            if math.isfinite(v) and c>0:
                vals.append(v); cnt.append(c)
    if not cnt: return float('nan'),0
    vals=np.asarray(vals,float); cnt=np.asarray(cnt,float)
    return float(np.sum(vals*cnt)/np.sum(cnt)), int(np.sum(cnt))

rows=[]
for tu in train_utils:
    for eu in eval_utils:
        for tk in (10,20):
            bar=os.path.join(out,f'proxy_{proxy}_trainu{tu}_eval{eu}_t{tk}_bar.csv')
            om,oc=wmean(bar,'orig_mean_fidelity','orig_count')
            nm,nc=wmean(bar,'new_mean_fidelity','new_count')
            rows.append({'train_util':tu,'eval_util':eu,'top':tk,'orig_mean':om,'new_mean':nm,'orig_count':oc,'new_count':nc})

# save long
with open(os.path.join(out,f'proxy_{proxy}_singleutil_physical_long.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['train_util','eval_util','top','orig_mean','new_mean','orig_count','new_count'])
    w.writeheader(); w.writerows(rows)

agg=[]
for tu in train_utils:
    sub=[r for r in rows if r['train_util']==tu]
    t10=[r for r in sub if r['top']==10]
    t20=[r for r in sub if r['top']==20]
    t10m=float(np.mean([r['new_mean'] for r in t10]))
    t20m=float(np.mean([r['new_mean'] for r in t20]))
    overall=float((t10m+t20m)/2.0)
    b10=float(np.mean([r['orig_mean'] for r in t10]))
    b20=float(np.mean([r['orig_mean'] for r in t20]))
    agg.append({'train_util':tu,'top10':t10m,'top20':t20m,'overall':overall,'baseline_top10':b10,'baseline_top20':b20,'baseline_overall':(b10+b20)/2.0})

with open(os.path.join(out,f'proxy_{proxy}_singleutil_physical_agg.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['train_util','top10','top20','overall','baseline_top10','baseline_top20','baseline_overall'])
    w.writeheader(); w.writerows(agg)

labels=[f'u{a["train_util"]}' for a in agg]
x=np.arange(len(agg)); width=0.35
v10=[a['top10'] for a in agg]; v20=[a['top20'] for a in agg]; vo=[a['overall'] for a in agg]
b10=float(np.mean([a['baseline_top10'] for a in agg]))
b20=float(np.mean([a['baseline_top20'] for a in agg]))
bo=float(np.mean([a['baseline_overall'] for a in agg]))

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5),dpi=160)
ax1.bar(x-width/2,v10,width,label='Top10%',color='#1f77b4')
ax1.bar(x+width/2,v20,width,label='Top20%',color='#ff7f0e')
ax1.axhline(b10,color='#1f77b4',ls='--',lw=1,alpha=0.8,label=f'Baseline Top10% ({b10:.3f})')
ax1.axhline(b20,color='#ff7f0e',ls='--',lw=1,alpha=0.8,label=f'Baseline Top20% ({b20:.3f})')
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylabel('Physical Mean Fidelity')
ax1.set_xlabel('Train-Util Method')
ax1.set_title('Physical Fidelity by Train-Util Method')
ax1.grid(axis='y',alpha=0.2)
ax1.legend(loc='upper right',fontsize=8)

ax2.bar(x,vo,width=0.6,color='#2ca02c',label='Mean(Top10%, Top20%)')
ax2.axhline(bo,color='gray',ls='--',lw=1,alpha=0.8,label=f'Baseline overall ({bo:.3f})')
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel('Mean(Top10%, Top20%)')
ax2.set_xlabel('Train-Util Method')
ax2.set_title('Overall Score (Higher Better)')
ax2.grid(axis='y',alpha=0.2)
ax2.legend(loc='upper right',fontsize=8)
for i,v in enumerate(vo):
    ax2.text(i,v+0.005,f'{v:.3f}',ha='center',va='bottom',fontsize=7)

fig.suptitle(f'Proxy({proxy}) evolved per train util, evaluated on Physical util30/60/88 (shots=8196)',fontsize=11)
fig.tight_layout()
png=os.path.join(out,f'proxy_{proxy}_trainutil_on_physical_summary.png')
pdf=os.path.join(out,f'proxy_{proxy}_trainutil_on_physical_summary.pdf')
fig.savefig(png); fig.savefig(pdf); plt.close(fig)
print('[OK]',png)
PY

done

echo "[DONE] postprocess for group ${GROUP_TAG}"

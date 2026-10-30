import re
lines = open('tmp/build_v.log', encoding='ascii', errors='replace').read().splitlines()
cur = None
for i, ln in enumerate(lines):
    m = re.search(r"Compiling entry function '([^']+)'", ln)
    if m:
        cur = m.group(1)
        continue
    m2 = re.search(r"Used (\d+) registers", ln)
    if m2 and cur:
        spill = 0
        if 'spill stores' in ln:
            mm = re.search(r"(\d+) bytes spill stores", ln)
            spill = int(mm.group(1)) if mm else 0
        else:
            for k in range(max(0, i - 2), i):
                if 'spill stores' in lines[k]:
                    mm = re.search(r"(\d+) bytes spill stores", lines[k])
                    spill = int(mm.group(1)) if mm else 0
        tag = cur
        if 'k_gemm_q5k' in tag or 'k_gemm_q6k' in tag or 'k_gemm_bf16' in tag or 'k_attn_tc' in tag:
            print(f"{m2.group(1):>4} regs  spill {spill:>4}  {tag[:80]}")
        cur = None

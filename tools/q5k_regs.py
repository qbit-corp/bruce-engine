import re

p = 'tmp/build_cp.log'
raw = open(p, 'rb').read()
if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
    s = raw.decode('utf-16')
else:
    s = raw.decode('utf-8', errors='replace')
print('len', len(s), '| has ptxas:', 'ptxas' in s)

blocks = re.findall(r"Compiling entry function '([^']+)'(.*?)(?=Compiling entry function|$)", s, re.S)
rows = []
for name, body in blocks:
    regs = re.findall(r"Used (\d+) registers", body)
    sp = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", body)
    if 'q5k' not in name and 'q6k' not in name:
        continue
    rows.append((name, regs[-1] if regs else '?', sp[-1] if sp else ('0', '0')))
for n, r, sp in sorted(rows):
    print('%-46s regs %-4s spill %s/%s' % (n[:46], r, sp[0], sp[1]))
print('q5k/q6k instantiations:', len(rows), 'of', len(blocks))

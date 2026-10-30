import re
s = open('tmp/build_cp.log', encoding='utf-8', errors='replace').read()
print('len', len(s), '| has ptxas:', 'ptxas' in s)
names = re.findall(r"Compiling entry function '([^']+)'", s)
print('entry functions:', len(names))
for n in names:
    if 'q5k' in n or 'q6k' in n:
        print(n)
print('--- first 6 raw lines ---')
for ln in s.splitlines()[:6]:
    print(ln[:160])

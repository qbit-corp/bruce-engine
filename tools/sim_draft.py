import subprocess, sys

ids = [int(x) for x in open('tmp/ids109.txt').read().replace(',', ' ').split()]
out = subprocess.run([r'engine\bruce.exe', '--ctx', '4096', '--token-file', 'tmp/ids109.txt',
                      '--gen', '96', '--greedy', '--print-ids'],
                     capture_output=True, text=True)
gen = [int(x) for x in out.stdout.split()]
print('gen', len(gen))

hist = list(ids)
tab = {g: {} for g in range(2, 5)}
dmax = 7
nmax, nmin = 4, 2


def hash_n(p):
    k = 1469598103934665603
    for v in p:
        k ^= (v & 0xFFFFFFFF)
        k = (k * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return k


def push(t):
    hist.append(t)
    n = len(hist)
    last = []
    for g in range(nmax, nmin - 1, -1):
        if last:
            break
        if n < g:
            continue
        it = tab[g].get(hash_n(hist[n - g:]))
        if it is None:
            continue
        j = it
        if j + 1 >= n:
            continue
        lastp = min(n - 1, j + dmax)
        last = hist[j + 1:lastp + 1]
    for g in range(nmin, nmax + 1):
        if n >= g:
            tab[g][hash_n(hist[n - g:])] = n - 1
    return last


for t in ids:
    push(t)
lens = {}
for t in gen:
    d = push(t)
    lens[len(d)] = lens.get(len(d), 0) + 1
print('draft length histogram (python sim):', dict(sorted(lens.items())))

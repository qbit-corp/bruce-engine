"""Compare the decode step (forward_launch) against an equivalent prefill chunk.

For a prefix of N tokens: run A prefills N tokens then decodes one token and prints
the resulting argmax; run B prefills N+1 tokens and prints the argmax.  If the decode
path is correct the two argmaxes are equal (up to chunk-semantics noise).
"""
import subprocess, sys

ENG = r'engine\bruce.exe'
ids = [int(x) for x in open('tmp/ids_code.txt').read().replace(',', ' ').split()]


def run(tokens, gen):
    args = [ENG, '--ctx', '4096', '--tokens', ','.join(map(str, tokens)),
            '--gen', str(gen), '--greedy', '--print-ids']
    r = subprocess.run(args, capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


for n in [3, 5, 9, 17, 33, 65, 121]:
    pre = ids[:n]
    a = run(pre, 2)                # [prefill argmax, decode-of-that argmax]
    b = run(pre + [a[0]], 1)       # prefill argmax of the extended prompt
    print('n=%3d  decode %d  prefill %d  %s' % (n, a[1], b[0], 'OK' if a[1] == b[0] else 'MISMATCH'))
    sys.stdout.flush()

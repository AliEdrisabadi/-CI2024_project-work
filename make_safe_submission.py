import re
import sys
from pathlib import Path

HEADER = """# Auto-sanitized wrapper
# (added by make_safe_submission.py)
EPS = 1e-9
def _sanitize(y):
    import numpy as np
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(y, -1e6, 1e6)
def pdiv(a, b):
    import numpy as np
    den = np.where(np.abs(b) < EPS, EPS, b)
    return a / den
def plog(a):
    import numpy as np
    return np.log(np.abs(a) + EPS)
def psqrt(a):
    import numpy as np
    return np.sqrt(np.abs(a))
def pexp(a):
    import numpy as np
    return np.exp(np.clip(a, -20.0, 20.0))
def ppow(a, b):
    import numpy as np
    return np.power(np.clip(a, -50.0, 50.0), np.clip(b, -3.0, 3.0))
"""

FUNC_RE = re.compile(
    r'(def\s+f([1-8])\s*\(\s*x:\s*np\.ndarray\s*\)\s*->\s*np\.ndarray\s*:\s*\n)'
    r'(\s*return\s+)([^\n]+)\n',
    flags=re.M | re.S
)

def main(inp: str, outp: str | None = None):
    src_path = Path(inp)
    text = src_path.read_text(encoding='utf-8')

    if '_sanitize(' not in text:
        # insert HEADER after first numpy import
        m = re.search(r'(^\s*import\s+numpy\s+as\s+np\s*$)', text, flags=re.M)
        if m:
            insert_at = m.end()
            text = text[:insert_at] + '\n\n' + HEADER + '\n' + text[insert_at:]
        else:
            
            text = 'import numpy as np\n\n' + HEADER + '\n' + text

    def _wrap(m):
        head, num, retkw, expr = m.groups()
        return f"{head}{retkw}_sanitize({expr})\n"

    text2, n = FUNC_RE.subn(_wrap, text)
    if n == 0:
        print("warning: no f1..f8 returns matched; file may already be sanitized.")
    out_path = Path(outp) if outp else src_path
    out_path.write_text(text2, encoding='utf-8')
    print(f"[ok] wrote {out_path} (wrapped {n} returns)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python make_safe_submission.py s316628.py [s316628_safe.py]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else None
    main(inp, outp)

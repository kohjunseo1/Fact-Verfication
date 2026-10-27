"""Helpers that reproduce JavaScript formatting/ordering so ported scripts print identical output."""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal

from title_utils import JS_WS, JS_WS_CLASS, js_trim  # noqa: F401  (re-exported)

# console.log writes '\n' on every platform; Python's stdout would write '\r\n' on Windows.
sys.stdout.reconfigure(newline="\n")


def js_sum(xs):
    """Left-to-right float addition like Array.reduce (Python >= 3.12 sum() compensates rounding)."""
    total = 0
    for x in xs:
        total += x
    return total


def js_sort_key(s: str) -> bytes:
    """Default Array.prototype.sort order for strings (UTF-16 code units)."""
    return s.encode("utf-16-be", "surrogatepass")


# JS '.' (no line terminators) for patterns that use it
JS_DOT = "[^\n\r" + chr(0x2028) + chr(0x2029) + "]"


def js_fixed(x: float, digits: int) -> str:
    """Number.prototype.toFixed: exact binary value, ties rounded up (Python rounds half-even)."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(x).quantize(q, rounding=ROUND_HALF_UP))


def _array_index(k: str) -> bool:
    return k.isdigit() and str(int(k)) == k and int(k) < 2 ** 32 - 1


def js_entries(d: dict):
    """Object.entries order: integer-like keys ascending, then the rest in insertion order."""
    items = [(str(k), v) for k, v in d.items()]
    ints = sorted((kv for kv in items if _array_index(kv[0])), key=lambda kv: int(kv[0]))
    return ints + [kv for kv in items if not _array_index(kv[0])]


def js_json(x) -> str:
    """JSON.stringify(x) without indentation."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def js_count(values):
    """cnt(): Object.entries of a count map, stably sorted by count descending."""
    m = {}
    for v in values:
        k = str(v)
        m[k] = m.get(k, 0) + 1
    return sorted(js_entries(m), key=lambda kv: -kv[1])


def js_str(x) -> str:
    """String(x) for console.log arguments."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else repr(x)
    return str(x)


def log(*args):
    """console.log: arguments joined by a space."""
    print(" ".join(js_str(a) for a in args))

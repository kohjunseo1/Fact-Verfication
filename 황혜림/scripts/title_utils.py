"""Shared Wikipedia title normalisation, line reading and 2017-10-01 redirect resolution."""
import codecs
import gzip
import re
import sys
from urllib.parse import unquote

# Writing a lone surrogate (possible after json.loads) as UTF-8: JS emits U+FFFD per code unit.
codecs.register_error("js_replace", lambda e: (chr(0xFFFD) * (e.end - e.start), e.end))


def open_out(path, mode="w"):
    """Text output with '\\n' newlines and JS-compatible handling of lone surrogates."""
    return open(path, mode, encoding="utf-8", errors="js_replace", newline="\n")

# Characters that must not appear inside a TSV field: tab, newline, CR,
# line/paragraph separators, NEL, VT, FF (built from code points on purpose).
_SEPARATORS = "".join(map(chr, (0x2028, 0x2029, 0x85, 0x0B, 0x0C)))
BREAKS = re.compile("[\t\n\r" + _SEPARATORS + "]")
BREAKS_NO_TAB = re.compile("[\r" + _SEPARATORS + "]")

# JavaScript's \s / String.prototype.trim() whitespace set (differs slightly from Python's).
JS_WS = "".join(map(chr, (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680, *range(0x2000, 0x200B),
                          0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF)))
JS_WS_CLASS = "[" + re.escape(JS_WS) + "]"
_WS = re.compile(JS_WS_CLASS + "+")


def js_trim(s: str) -> str:
    return s.strip(JS_WS)


def norm_title(t: str) -> str:
    """MediaWiki canonical form: underscores -> spaces, collapse whitespace,
    drop #section, upper-case first character."""
    t = js_trim(_WS.sub(" ", t.replace("_", " ")))
    h = t.find("#")
    if h >= 0:
        t = js_trim(t[:h])
    if not t:
        return ""
    if ord(t[0]) > 0xFFFF:   # JS t[0] is a lone surrogate here and is left unchanged
        return t
    return t[0].upper() + t[1:]


def read_lines(source):
    """Yield lines split on '\\n' only (str.splitlines/text mode would also split
    on U+2028, U+2029 or \\r, which occur inside some titles and anchors).
    source: path (.gz is decompressed), or a binary stream such as sys.stdin.buffer."""
    if isinstance(source, str) or hasattr(source, "__fspath__"):
        path = str(source)
        f = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
        close = True
    else:
        f, close = source, False
    try:
        for raw in f:
            yield raw.decode("utf-8", errors="replace").removesuffix("\n")
    finally:
        if close:
            f.close()


def stdin_lines():
    return read_lines(sys.stdin.buffer)


def load_redirects(pages_tsv, on_page=None):
    """pages_ns0 tsv: page_id \\t title \\t redirect_target.
    Returns {title: target}; on_page(id, title, target) is called for every row."""
    redirects = {}
    for line in read_lines(pages_tsv):
        a = line.find("\t")
        b = line.find("\t", a + 1)
        if b < 0:
            continue
        title, target = line[a + 1:b], line[b + 1:]
        if target:
            redirects[title] = target
        if on_page:
            on_page(line[:a], title, target)
    return redirects


_ENTITY = re.compile(r"&(#x[0-9a-f]+|#\d+|[a-z]+);", re.IGNORECASE)
_NAMED = {"amp": "&", "quot": '"', "apos": "'", "lt": "<", "gt": ">", "nbsp": " ", "ndash": "–", "mdash": "—"}


def html_unescape(s: str) -> str:
    """Decode the HTML entities left in hrefs/anchors (&amp; &nbsp; &#160; &#x27; ...).
    Must run before '#' is treated as a section separator."""
    def sub(m):
        e = m.group(1)
        if e[0] == "#":
            cp = int(e[2:], 16) if e[1] in "xX" else int(e[1:])
            if cp == 160:
                return " "
            return chr(cp) if 0 <= cp <= 0x10FFFF else m.group(0)
        return _NAMED.get(e.lower(), m.group(0))
    return _ENTITY.sub(sub, s)


_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


def decode_uri_component(s: str) -> str:
    """JS decodeURIComponent; returns s unchanged where JS would throw."""
    if _BAD_PERCENT.search(s):
        return s
    try:
        return unquote(s, errors="strict")
    except UnicodeDecodeError:
        return s


def canon_map(titles):
    """HotpotQA/HoVer titles keep HTML escapes ("Sugar &amp; Spice"), while hrefs and the
    Wikipedia redirect table do not. Map unescaped form -> stored node title."""
    return {html_unescape(t): t for t in titles}


def resolve(raw: str, canon: dict, redirects: dict):
    """Resolve a raw link target to a stored node title.
    Returns (kind, title); kind: direct | case_fixed | via_redirect | unresolved | empty."""
    exact = js_trim(js_trim(_WS.sub(" ", html_unescape(raw).replace("_", " "))).split("#")[0])
    if not exact:
        return "empty", None
    if exact in canon:
        return "direct", canon[exact]
    t = norm_title(exact)
    if t in canon:
        return "case_fixed", canon[t]
    hop = 0
    while hop < 5 and t in redirects:
        t = norm_title(redirects[t])
        if t in canon:
            return "via_redirect", canon[t]
        hop += 1
    return "unresolved", norm_title(exact)

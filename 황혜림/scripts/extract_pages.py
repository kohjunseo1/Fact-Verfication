"""Stream a MediaWiki pages-articles XML dump from stdin and write the namespace-0
page table: page_id \\t title \\t redirect_target ('' if not a redirect).

Usage: curl ... | bzip2 -dc | python extract_pages.py out.tsv
"""
import sys

from title_utils import open_out, stdin_lines


def unescape(s: str) -> str:
    return (s.replace("&quot;", '"').replace("&#039;", "'").replace("&lt;", "<")
             .replace("&gt;", ">").replace("&amp;", "&"))


def main(out_path: str):
    title = ns = page_id = None
    redirect = ""
    pages = redirects = 0
    with open_out(out_path) as out:
        for line in stdin_lines():
            t = line.lstrip()
            if t.startswith("<page>"):
                title = ns = page_id = None
                redirect = ""
            elif t.startswith("<title>"):
                title = unescape(t[7:t.find("</title>")])
            elif t.startswith("<ns>"):
                ns = t[4:t.find("</ns>")]
            elif page_id is None and t.startswith("<id>"):
                page_id = t[4:t.find("</id>")]
            elif t.startswith('<redirect title="'):
                redirect = unescape(t[17:t.find('"', 17)])
            elif t.startswith("</page>") and ns == "0":
                out.write(f"{page_id}\t{title}\t{redirect}\n")
                pages += 1
                redirects += bool(redirect)
                if pages % 1_000_000 == 0:
                    print(f"{pages} ns0 pages ({redirects} redirects)", file=sys.stderr)
    print(f"done: {pages} ns0 pages, {redirects} redirects", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])

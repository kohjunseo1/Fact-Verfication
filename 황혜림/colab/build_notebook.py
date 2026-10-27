"""Convert a `# %%` percent-format script into a Jupyter notebook.

Usage: python build_notebook.py hover_retrieval.py hover_retrieval.ipynb
"""
import json
import sys


def cells(src: str):
    kind, buf = None, []
    for line in src.split(chr(10)):   # not splitlines(): it also breaks on U+2028 etc.
        if line.startswith("# %%"):
            if kind is not None:
                yield kind, buf
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
            continue
        if kind == "markdown":
            line = line[2:] if line.startswith("# ") else line.lstrip("#")
        buf.append(line)
    if kind is not None:
        yield kind, buf


def main(src_path: str, out_path: str):
    nb_cells = []
    for kind, lines in cells(open(src_path, encoding="utf-8").read()):
        while lines and not lines[-1].strip():
            lines.pop()
        source = [l + "\n" for l in lines[:-1]] + lines[-1:]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    nb = {
        "cells": nb_cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{len(nb_cells)} cells -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])

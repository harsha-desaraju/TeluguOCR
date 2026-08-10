"""Inline a training loop and everything it imports into ONE standalone .py file.

WHY THIS EXISTS
    CLAUDE.md's rule was that training scripts inline copies of the model, tokenizer and
    utils so each file can be uploaded to Kaggle / Colab and run with nothing else
    present. That constraint is real -- those hosts get one file, not a package.

    The convention was sound; HAND-COPYING was the problem. By the time phase 3 started
    there were 69 duplicated top-level definitions across the repo and 41 of them had
    drifted apart, including four versions of MultiHeadAttention and seven ways to
    preprocess a line image. Phase 3 collapsed them onto one copy each, which cost the
    loops their standalone property.

    This script gives it back, without the drift: the loops import from the package
    like normal code, and the single-file artifact is GENERATED. There is exactly one
    definition of everything, and the copy that lands on Kaggle is derived from it
    rather than maintained beside it.

USAGE
    python3 scripts/bundle.py src/telugu_ocr/training/loops/encdec.py -o encdec_kaggle.py
    python3 scripts/bundle.py src/telugu_ocr/training/loops/ctc.py            # -> stdout path

    Then upload the emitted file. It imports only third-party packages.

HOW IT WORKS
    Walk the entry module's `src.telugu_ocr.*` imports transitively, topologically sort
    the resulting modules so a definition always precedes its use, strip the first-party
    import statements, hoist the third-party ones, and concatenate.

    The one subtlety is ORDER AT THE TOP. The loops set OMP/MKL thread-limit environment
    variables that only take effect if they are set BEFORE numpy and cv2 are imported.
    So the entry module's prologue -- everything above its first package import -- is
    emitted first, verbatim, and the hoisted imports come after it. Getting that
    backwards would not fail; it would just silently spawn one BLAS thread per core
    inside every dataloader worker and thrash.

WHAT IT DOES NOT DO
    No dead-code elimination: if a bundled module defines something the loop never
    calls, it still lands in the file. That is deliberate -- tracking usage through
    getattr and class hierarchies to decide what is safe to drop is exactly the kind of
    cleverness that produces a file which imports fine and breaks at step 3000.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIRST_PARTY_PREFIX = "src.telugu_ocr"


def module_to_path(dotted: str) -> pathlib.Path | None:
    p = REPO_ROOT.joinpath(*dotted.split("."))
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    return None


def path_to_module(path: pathlib.Path) -> str:
    rel = path.resolve().relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def first_party_imports(tree: ast.Module) -> list[str]:
    """Dotted first-party module names this module imports, in source order."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith(FIRST_PARTY_PREFIX):
                out.append(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(FIRST_PARTY_PREFIX):
                    out.append(a.name)
    return out


def is_first_party_import(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return node.module.startswith(FIRST_PARTY_PREFIX)
    if isinstance(node, ast.Import):
        return any(a.name.startswith(FIRST_PARTY_PREFIX) for a in node.names)
    return False


def collect(entry: str, seen: set[str], order: list[str]) -> None:
    """Depth-first post-order: a module is appended after everything it imports."""
    if entry in seen:
        return
    seen.add(entry)
    path = module_to_path(entry)
    if path is None:
        return
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for dep in first_party_imports(tree):
        collect(dep, seen, order)
    order.append(entry)


def split_module(path: pathlib.Path, is_entry: bool):
    """-> (prologue_src, future_imports, third_party_imports, body_src)."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)

    futures, imports, body_nodes = [], [], []
    prologue_end = 0
    if is_entry:
        # Everything above the first first-party import runs before the hoisted imports,
        # so the loops' OMP/MKL env-var block still lands before numpy and cv2.
        for node in tree.body:
            if is_first_party_import(node):
                break
            prologue_end = node.end_lineno

    for node in tree.body:
        if is_entry and node.end_lineno <= prologue_end:
            continue
        if is_first_party_import(node):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(ast.unparse(node))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
            continue
        body_nodes.append(node)

    prologue = ""
    if is_entry:
        # __future__ imports are hoisted to the very top of the bundle, so drop them from
        # the prologue -- two copies, with the second below other statements, is a
        # SyntaxError that ast.parse happily accepts and compile() rejects.
        fut_lines = {i for n in tree.body
                     if isinstance(n, ast.ImportFrom) and n.module == "__future__"
                     for i in range(n.lineno, n.end_lineno + 1)}
        prologue = "".join(l for i, l in enumerate(lines[:prologue_end], 1)
                           if i not in fut_lines)
    if body_nodes:
        start = min(min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
                    for n in body_nodes)
        end = max(n.end_lineno for n in body_nodes)
        # Drop every TOP-LEVEL import line inside the body span. Not just the
        # first-party ones: a module whose docstring precedes its imports makes the span
        # start at line 1, so the __future__ and third-party lines fall inside it too --
        # and those are emitted separately at the top. Imports nested inside functions
        # are untouched, since this only walks tree.body.
        drop = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                drop.update(range(node.lineno, node.end_lineno + 1))
        body = "".join(l for i, l in enumerate(lines[start - 1:end], start) if i not in drop)
    else:
        body = ""
    return prologue, futures, imports, body


def bundle(entry_path: pathlib.Path) -> str:
    entry_mod = path_to_module(entry_path)
    order: list[str] = []
    collect(entry_mod, set(), order)
    deps = [m for m in order if m != entry_mod]

    prologue, futures, imports, entry_body = split_module(entry_path, is_entry=True)

    dep_chunks, seen_imports, seen_futures = [], set(), set()
    # The entry module's own imports BELOW its first package import must be emitted too,
    # not merely deduped against. ctc.py keeps `import cv2` down beside the augmentation
    # section; treating that as already-present produced a bundle that imported fine
    # until the first cv2 call.
    hoisted = []
    for name in imports:
        if name not in seen_imports:
            seen_imports.add(name)
            hoisted.append(name)
    for f in futures:
        seen_futures.add(f)
    for mod in deps:
        path = module_to_path(mod)
        if path is None:
            continue
        _, dfut, dimp, dbody = split_module(path, is_entry=False)
        for f in dfut:
            if f not in seen_futures:
                seen_futures.add(f)
                futures.append(f)
        for i in dimp:
            if i not in seen_imports:
                seen_imports.add(i)
                hoisted.append(i)
        if dbody.strip():
            dep_chunks.append((mod, dbody))

    out = [
        '"""GENERATED FILE -- do not edit.\n',
        f"Bundled from {entry_path.relative_to(REPO_ROOT)} by scripts/bundle.py.\n",
        "\nEvery definition below is inlined from the telugu_ocr package, in dependency\n",
        "order, so this file runs standalone on Kaggle / Colab with only third-party\n",
        "packages installed. Edit the SOURCE modules and re-bundle; edits here are lost.\n",
        "\nInlined modules, in order:\n",
    ]
    out += [f"    {m}\n" for m in deps]
    out += [f"    {entry_mod}   (entry)\n", '"""\n']

    # __future__ imports must precede every other statement.
    if futures:
        out.append("\n" + "\n".join(sorted(set(futures))) + "\n")
    if prologue.strip():
        out.append("\n# " + "=" * 84 + "\n")
        out.append(f"# Prologue from {entry_mod} -- runs BEFORE the hoisted imports, because the\n")
        out.append("# thread-limit environment variables only bind if they are set before numpy\n")
        out.append("# and cv2 are first imported.\n")
        out.append("# " + "=" * 84 + "\n")
        out.append(prologue.rstrip("\n") + "\n")
    if hoisted:
        out.append("\n# " + "=" * 84 + "\n# Hoisted third-party imports\n# " + "=" * 84 + "\n")
        out.append("\n".join(hoisted) + "\n")
    for mod, body in dep_chunks:
        out.append("\n\n# " + "=" * 84 + f"\n# inlined from {mod}\n# " + "=" * 84 + "\n")
        out.append(body.rstrip("\n") + "\n")
    out.append("\n\n# " + "=" * 84 + f"\n# entry: {entry_mod}\n# " + "=" * 84 + "\n")
    out.append(entry_body.rstrip("\n") + "\n")

    text = "".join(out)
    # compile(), not ast.parse(): ast.parse accepts a misplaced `from __future__` that
    # the real compiler rejects, so parsing alone would have shipped a broken bundle.
    compile(text, "<bundle>", "exec")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("entry", help="module to bundle, e.g. src/telugu_ocr/training/loops/encdec.py")
    ap.add_argument("-o", "--out", help="output path (default: <name>_bundled.py in cwd)")
    args = ap.parse_args()

    entry_path = pathlib.Path(args.entry).resolve()
    if not entry_path.is_file():
        print(f"no such file: {entry_path}", file=sys.stderr)
        return 2

    text = bundle(entry_path)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(f"{entry_path.stem}_bundled.py")
    out.write_text(text, encoding="utf-8")

    n_defs = sum(isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 for n in ast.parse(text).body)
    print(f"{entry_path.relative_to(REPO_ROOT)} -> {out}  "
          f"({len(text.splitlines())} lines, {n_defs} top-level defs, no first-party imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public API symbol-set snapshot for camxiom (roadmap Rel1a).

Extract the public symbol surface from the camxiom public headers
(``include/camxiom/**``, excluding ``internal/``) and compare it against a
checked-in golden snapshot, failing on any drift. This is the guardrail that
detects *unintended* public API changes (e.g. the #3 removal of the per-model
namespaces) before the ``0.1.0`` release is sealed.

Design notes
------------
* **Config-independent by construction.** The extractor parses header *source
  text*, not compiled symbols. It therefore produces byte-identical output
  regardless of whether Ceres / OpenCV / ROS were found at build time, and
  regardless of host architecture (aarch64 dev box vs x86 CI). ``nm`` on the
  built libraries would drift with build flags, optimisation level, compiler
  version and template instantiation -- unusable as a stable snapshot.
* **What it captures:** namespace markers, class/struct/union/enum markers,
  public member functions/variables, free functions, ``using`` aliases /
  re-exports, and enumerators (name + explicit value, in source order). This
  catches additions, removals, renames, signature changes, namespace add/remove,
  public-header add/remove, and enumerator add/remove/reorder.
* **Deliberately out of scope:** private/protected members, ``internal/``
  headers, function bodies, default-initialiser *values*, and ``static_assert``.
  Those are guarded by the behavioural tests, not by an API-surface snapshot.

Usage
-----
    check_public_api.py --include-dir <dir> --snapshot <file> [--update]

``--update`` (or the env var ``CAMXIOM_UPDATE_API_SNAPSHOT=1``) regenerates the
golden after a *deliberate* API change; otherwise a mismatch prints a unified
diff and exits non-zero.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source cleaning: strip comments, preprocessor lines, and blank out string /
# char literals so the structural scan below only sees code punctuation.
# ---------------------------------------------------------------------------


def clean_source(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            out.append(" ")
            continue
        if c == '"':
            out.append('""')
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if c == "'":
            out.append("''")
            i += 1
            while i < n and text[i] != "'":
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        out.append(c)
        i += 1
    cleaned = "".join(out)

    # Drop preprocessor lines (keep the code they guard). Handle backslash
    # line-continuations so a multi-line macro is fully removed.
    kept = []
    skip_continuation = False
    for line in cleaned.split("\n"):
        if skip_continuation:
            skip_continuation = line.rstrip().endswith("\\")
            continue
        if line.lstrip().startswith("#"):
            skip_continuation = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Head classification: decide whether a token run before a '{' opens a scope
# (namespace / class / struct / union / enum) and extract its name.
# ---------------------------------------------------------------------------

_QUALIFIERS = ("constexpr", "inline", "static", "friend", "explicit", "virtual", "typedef")


def _strip_template_and_qualifiers(head: str) -> str:
    h = head.strip()
    changed = True
    while changed:
        changed = False
        if h.startswith("template"):
            rest = h[len("template"):].lstrip()
            if rest.startswith("<"):
                depth = 0
                for j, ch in enumerate(rest):
                    if ch == "<":
                        depth += 1
                    elif ch == ">":
                        depth -= 1
                        if depth == 0:
                            h = rest[j + 1:].lstrip()
                            changed = True
                            break
                if changed:
                    continue
        if h.startswith("[["):
            k = h.find("]]")
            if k != -1:
                h = h[k + 2:].lstrip()
                changed = True
                continue
        for q in _QUALIFIERS:
            if re.match(r"^" + q + r"\b", h):
                h = h[len(q):].lstrip()
                changed = True
                break
    return h


def classify_head(head: str):
    """Return (kind, name) if `head` opens a scope, else ('', '')."""
    h = _strip_template_and_qualifiers(head)
    m = re.match(r"^namespace\b\s*(.*)$", h)
    if m:
        return ("namespace", m.group(1).strip())
    # Attributes may legally appear after the enum-key ("enum class
    # [[nodiscard]] Name"); skip them so the enum NAME is captured, not the
    # literal token "class".
    m = re.match(r"^enum\b\s*(?:class|struct)?\s*(?:\[\[[^\]]*\]\]\s*)*([A-Za-z_]\w*)", h)
    if m:
        return ("enum", m.group(1))
    m = re.match(r"^(class|struct|union)\b\s*(?:alignas\s*\([^)]*\)\s*)?([A-Za-z_]\w*)", h)
    if m:
        return (m.group(1), m.group(2))
    return ("", "")


def normalize(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Scope-tracking extractor.
# ---------------------------------------------------------------------------


class Frame:
    __slots__ = ("kind", "name", "access", "suppressed")

    def __init__(self, kind, name, access, suppressed):
        self.kind = kind          # file / namespace / class / struct / union / enum
        self.name = name
        self.access = access      # public / private / protected
        self.suppressed = suppressed


_SCOPE_KINDS = ("namespace", "class", "struct", "union", "enum")


def _skip_braces(s: str, i: int) -> int:
    """`s[i]` is '{'; return the index just past the matching '}'."""
    depth = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def extract_header(cleaned: str) -> list:
    stack = [Frame("file", "", "public", False)]
    lines = []
    buf = []
    paren = 0
    i = 0
    n = len(cleaned)

    def qual_of() -> str:
        return "::".join(f.name for f in stack if f.kind in _SCOPE_KINDS and f.name)

    def emit(qual: str, text: str):
        text = normalize(text)
        if text:
            lines.append(f"{qual} | {text}" if qual else text)

    def cur_suppressed() -> bool:
        f = stack[-1]
        if f.suppressed:
            return True
        if f.kind in ("class", "struct", "union") and f.access != "public":
            return True
        return False

    while i < n:
        f = stack[-1]
        c = cleaned[i]

        # --- enum body: enumerators separated by top-level commas -----------
        if f.kind == "enum":
            if c == "(":
                paren += 1
                buf.append(c)
                i += 1
                continue
            if c == ")":
                paren -= 1
                buf.append(c)
                i += 1
                continue
            if c == "}" and paren == 0:
                if not f.suppressed:
                    ename = "".join(buf).strip()
                    if ename:
                        idx = sum(1 for ln in lines if ln.startswith(qual_of() + " | #"))
                        emit(qual_of(), f"#{idx:03d} {ename}")
                buf = []
                stack.pop()
                i += 1
                continue
            if c == "," and paren == 0:
                if not f.suppressed:
                    ename = "".join(buf).strip()
                    if ename:
                        idx = sum(1 for ln in lines if ln.startswith(qual_of() + " | #"))
                        emit(qual_of(), f"#{idx:03d} {ename}")
                buf = []
                i += 1
                continue
            buf.append(c)
            i += 1
            continue

        # --- normal scope ---------------------------------------------------
        if c == "(":
            paren += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            paren -= 1
            buf.append(c)
            i += 1
            continue

        if c == ":" and paren == 0 and f.kind in ("class", "struct", "union"):
            label = "".join(buf).strip()
            nxt = cleaned[i + 1] if i + 1 < n else ""
            if label in ("public", "private", "protected") and nxt != ":":
                f.access = label
                buf = []
                i += 1
                continue
            buf.append(c)
            i += 1
            continue

        if c == "{":
            if paren > 0:
                buf.append(c)
                i += 1
                continue
            head = "".join(buf).strip()
            kind, name = classify_head(head)
            if kind:
                parent_suppressed = cur_suppressed()
                suppressed = parent_suppressed
                if kind == "namespace":
                    segs = [s for s in re.split(r"::|\s", name) if s]
                    if any(s in ("internal", "detail") for s in segs):
                        suppressed = True
                if not suppressed:
                    emit(qual_of(), head)
                default_access = "private" if kind == "class" else "public"
                stack.append(Frame(kind, name, default_access, suppressed))
                buf = []
                i += 1
                continue
            # function body or braced initialiser
            if "(" in head and ")" in head:
                if not cur_suppressed():
                    emit(qual_of(), head)
                i = _skip_braces(cleaned, i)
                buf = []
                # swallow an optional trailing ';'
                j = i
                while j < n and cleaned[j] in " \t\r\n":
                    j += 1
                if j < n and cleaned[j] == ";":
                    i = j + 1
                continue
            # braced initialiser for a variable: drop the initialiser, keep the
            # declarator, finalise at the ';'.
            i = _skip_braces(cleaned, i)
            continue

        if c == ";":
            if paren == 0:
                head = "".join(buf).strip()
                if head and not head.startswith("static_assert") and not cur_suppressed():
                    emit(qual_of(), head)
                buf = []
                i += 1
                continue
            buf.append(c)
            i += 1
            continue

        if c == "}":
            if paren > 0:
                buf.append(c)
                i += 1
                continue
            if len(stack) > 1:
                stack.pop()
            buf = []
            i += 1
            continue

        buf.append(c)
        i += 1

    return sorted(set(lines))


# ---------------------------------------------------------------------------
# Snapshot rendering.
# ---------------------------------------------------------------------------

_BANNER = [
    "# camxiom public API symbol-set snapshot (roadmap Rel1a).",
    "# AUTO-GENERATED by tests/api_snapshot/check_public_api.py -- do not edit by hand.",
    "# Regenerate after a DELIBERATE API change:",
    "#   CAMXIOM_UPDATE_API_SNAPSHOT=1 ctest -R camxiom_api_snapshot_test",
    "#   (or run the script directly with --update)",
    "# A diff here means the public API surface changed. If unintended, revert it;",
    "# if intended, review the diff and regenerate this file in the same commit.",
]


def collect_headers(include_dir: Path) -> list:
    root = include_dir / "camxiom"
    headers = []
    for p in sorted(root.rglob("*.hpp")):
        rel_parts = p.relative_to(root).parts
        if "internal" in rel_parts:
            continue
        headers.append(p)
    return headers


def render_snapshot(include_dir: Path) -> str:
    pkg_root = include_dir.parent
    sections = []
    for hdr in collect_headers(include_dir):
        rel = hdr.relative_to(pkg_root).as_posix()
        cleaned = clean_source(hdr.read_text(encoding="utf-8"))
        entries = extract_header(cleaned)
        block = [f"=== {rel} ==="]
        block.extend(entries)
        sections.append("\n".join(block))
    body = "\n\n".join(sections)
    return "\n".join(_BANNER) + "\n\n" + body + "\n"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="camxiom public API snapshot check")
    parser.add_argument("--include-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--update", action="store_true",
                        help="regenerate the golden snapshot instead of checking")
    args = parser.parse_args(argv)

    include_dir = args.include_dir.resolve()
    if not (include_dir / "camxiom").is_dir():
        print(f"ERROR: {include_dir}/camxiom is not a directory", file=sys.stderr)
        return 2

    current = render_snapshot(include_dir)

    update = args.update or os.environ.get("CAMXIOM_UPDATE_API_SNAPSHOT", "") not in ("", "0", "false", "False")

    if update:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(current, encoding="utf-8")
        print(f"camxiom API snapshot updated: {args.snapshot}")
        return 0

    if not args.snapshot.is_file():
        print(f"ERROR: snapshot file not found: {args.snapshot}\n"
              f"       generate it with --update (or CAMXIOM_UPDATE_API_SNAPSHOT=1).",
              file=sys.stderr)
        return 2

    golden = args.snapshot.read_text(encoding="utf-8")
    if golden == current:
        print("camxiom public API snapshot OK (no drift).")
        return 0

    diff = difflib.unified_diff(
        golden.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile=f"{args.snapshot.name} (golden)",
        tofile="current headers",
    )
    sys.stdout.write("".join(diff))
    print(
        "\nERROR: camxiom public API surface changed vs the golden snapshot.\n"
        "  * If this change is UNINTENDED, revert it.\n"
        "  * If it is INTENDED, review the diff and regenerate the golden in the\n"
        "    same commit: CAMXIOM_UPDATE_API_SNAPSHOT=1 ctest -R camxiom_api_snapshot_test\n"
        "    (or run tests/api_snapshot/check_public_api.py --update).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

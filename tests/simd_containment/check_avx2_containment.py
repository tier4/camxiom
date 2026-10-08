#!/usr/bin/env python3
# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that the AVX2 kernels are compiled in and stay confined to themselves.

Runtime SIMD dispatch (src/detail/simd_dispatch.hpp) compiles the AVX2 kernels
into every x86 build and guards them with cpuHasAvx2(). That arrangement has
exactly two ways to go wrong, and both are silent:

  * the kernels are not actually compiled -- the dispatch never fires and the
    library quietly runs SSE2 everywhere, which is the bug runtime dispatch was
    introduced to fix;
  * AVX2 instructions leak out of the target("avx2,fma") functions into code
    that runs unconditionally -- the library then executes an illegal
    instruction on any CPU without AVX2, and nothing on an AVX2 development
    machine or CI runner would ever notice.

Neither shows up in a functional test on an AVX2 host, so this inspects the
built archive directly. It reads the disassembly, so it needs objdump; where
objdump is missing or the target is not x86 the check reports SKIPPED.

Usage: check_avx2_containment.py <path-to-libcamxiom_core.a>
"""

import argparse
import re
import shutil
import subprocess
import sys

# 256-bit register operands and the FMA / AVX2-only mnemonics. Matching on ymm
# alone would miss nothing in practice, but the mnemonics make a leak easier to
# read in the failure output.
AVX_INSN = re.compile(r"%ymm\d+|\bv?f(?:n?m(?:add|sub))\w*\b|\bvpbroadcast|\bvperm")

# Every function that is allowed to contain them carries Avx in its name: the
# kernels themselves (rayToPixel*Avx8, pixelToRay*Avx4/8) and the batch entry
# points (*BatchAvx2*). The naming convention is what makes this check cheap;
# a new AVX2 function that does not follow it will be reported as a leak.
AVX_FUNC_NAME = re.compile(r"Avx")

FUNC_HEADER = re.compile(r"^[0-9a-f]+ <(.+)>:$")


def disassemble(archive: str) -> str:
    return subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", "-C", archive],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def scan(disasm: str):
    """Return (avx_functions, leaks) found in the disassembly."""
    current = None
    avx_functions = set()
    leaks: dict[str, list[str]] = {}

    for line in disasm.splitlines():
        header = FUNC_HEADER.match(line)
        if header:
            current = header.group(1)
            continue
        if current is None or not line.startswith(" "):
            continue
        if not AVX_INSN.search(line):
            continue
        if AVX_FUNC_NAME.search(current):
            avx_functions.add(current)
        else:
            leaks.setdefault(current, []).append(line.strip())

    return avx_functions, leaks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to libcamxiom_core.a")
    args = parser.parse_args()

    if shutil.which("objdump") is None:
        print("SKIPPED: objdump not available")
        return 0

    disasm = disassemble(args.archive)

    # Non-x86 targets (the aarch64 dev machine) compile no AVX2 at all, and the
    # NEON path is covered by the functional tests instead.
    if "%rsp" not in disasm and "%esp" not in disasm:
        print("SKIPPED: archive is not x86")
        return 0

    avx_functions, leaks = scan(disasm)

    if not avx_functions:
        print(
            "FAILED: no AVX2 instructions found in any *Avx* function.\n"
            "        The AVX2 kernels did not make it into the build, so\n"
            "        cpuHasAvx2() can only ever select dead code."
        )
        return 1

    if leaks:
        print(f"FAILED: AVX2 instructions in {len(leaks)} function(s) without a target attribute.")
        print("        These execute unconditionally and will fault on a CPU without AVX2.")
        for name, instructions in sorted(leaks.items()):
            print(f"  {name}")
            for instruction in instructions[:3]:
                print(f"      {instruction}")
        return 1

    print(f"PASSED: {len(avx_functions)} AVX2 function(s) compiled, none leaking into baseline code")
    return 0


if __name__ == "__main__":
    sys.exit(main())

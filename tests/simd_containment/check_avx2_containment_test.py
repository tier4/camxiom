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

"""Unit tests for GNU objdump selection in the AVX2 containment check."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest import mock


_CHECK_PATH = Path(__file__).with_name("check_avx2_containment.py")
_SPEC = importlib.util.spec_from_file_location("check_avx2_containment", _CHECK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
check_avx2_containment = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_avx2_containment)


class FindGnuObjdumpTest(unittest.TestCase):
    def test_accepts_gnu_objdump(self) -> None:
        result = subprocess.CompletedProcess(
            args=["/usr/bin/objdump", "--version"],
            returncode=0,
            stdout="GNU objdump (GNU Binutils) 2.42\n",
            stderr="",
        )
        with mock.patch.object(check_avx2_containment.shutil, "which", return_value="/usr/bin/objdump"):
            with mock.patch.object(check_avx2_containment.subprocess, "run", return_value=result):
                self.assertEqual(check_avx2_containment.find_gnu_objdump(), "/usr/bin/objdump")

    def test_skips_non_gnu_objdump(self) -> None:
        result = subprocess.CompletedProcess(
            args=["/opt/go/bin/objdump", "--version"],
            returncode=2,
            stdout="",
            stderr="flag provided but not defined: -version\n",
        )
        output = io.StringIO()
        with mock.patch.object(check_avx2_containment.shutil, "which", return_value="/opt/go/bin/objdump"):
            with mock.patch.object(check_avx2_containment.subprocess, "run", return_value=result):
                with contextlib.redirect_stdout(output):
                    self.assertIsNone(check_avx2_containment.find_gnu_objdump())
        self.assertIn("SKIPPED: GNU objdump required", output.getvalue())

    def test_skips_missing_objdump(self) -> None:
        output = io.StringIO()
        with mock.patch.object(check_avx2_containment.shutil, "which", return_value=None):
            with contextlib.redirect_stdout(output):
                self.assertIsNone(check_avx2_containment.find_gnu_objdump())
        self.assertEqual("SKIPPED: GNU objdump not available\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()

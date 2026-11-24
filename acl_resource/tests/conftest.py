"""Test fixtures.  Tests run on the made-up corpus only.

ACL_TEST_ROOT   folder with the layout written by ``acl_resource.synthetic.make_synthetic`` (optional: with the
                reference packages beside it, for the parity tests).  Unset: the corpus is generated once per
                test session into a temporary folder.
ACL_TEST_CORPUS processed corpus to test against (default: ACL_TEST_ROOT/processed/local).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True


@pytest.fixture(scope="session")
def test_root(tmp_path_factory):
    given = os.environ.get("ACL_TEST_ROOT")
    if given:
        return Path(given).resolve()
    from acl_resource.synthetic import make_synthetic
    root = tmp_path_factory.mktemp("made_up_corpus")
    make_synthetic.make(root / "corpus")
    return root / "corpus"


@pytest.fixture(scope="session")
def corpus(test_root):
    given = os.environ.get("ACL_TEST_CORPUS")
    return Path(given).resolve() if given else test_root / "processed" / "local"


@pytest.fixture(scope="session")
def planted(test_root):
    path = test_root / "PLANTED.json"
    if not path.is_file():
        pytest.skip("no planted structure in this test root")
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def reference(test_root):
    """Imports of the reference packages, or a skip when they are not beside the test root."""
    if not (test_root / "ensemble_round2" / "protocol.py").is_file():
        pytest.skip("reference packages are not available")
    if str(test_root) not in sys.path:
        sys.path.insert(0, str(test_root))
    import importlib
    return importlib.import_module("ensemble_round2.protocol")

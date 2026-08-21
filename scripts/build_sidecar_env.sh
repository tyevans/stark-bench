#!/usr/bin/env bash
# Build the Python 3.11 environment the scoring sidecar runs in.
#
# Scoring shells out to a 3.11 interpreter with `stark-qa` installed --
# 3.11 because `stark_qa` pulls `ogb` -> `rdkit`, and `numpy<2` because
# rdkit crashes on NumPy 2.x. Without this, every scoring run resolves 166
# packages from PyPI. Warm that is ~114ms; cold, or with PyPI degraded, it
# is a hard failure AFTER all retrieval has been paid for.
#
# Optional. `stark_scorer` falls back to `uv run --with` when this does not
# exist, so a fresh checkout still works -- see `_sidecar_command`.
#
# Re-run it to pick up a newer `stark-qa`: the prebuilt environment pins
# whatever it was built with, which is the point and also the trap.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".sidecar-venv"
echo "building $VENV (python 3.11, stark-qa, numpy<2)"
uv venv --python 3.11 "$VENV"
# `setuptools` is not optional and is not a dependency of stark-qa.
# `tdc.metadata` imports `pkg_resources`, which uv's ephemeral `--with`
# environment happens to provide and a fresh `uv venv` does not -- so the
# prebuilt path fails on an import the resolved path survives. Caught by
# the verification below on the first build, which is why it is here.
# `setuptools<81` is not cosmetic. `tdc.metadata` -- reached from
# `stark_qa.evaluator` -- imports `pkg_resources`, which setuptools REMOVED
# in v81. uv's ephemeral `--with` environment ships `pkg_resources`
# regardless, so the resolved path survives an import the prebuilt path
# fails on. Caught by the verification below on the first build, which is
# the reason it is here rather than in a comment.
uv pip install --python "$VENV/bin/python" stark-qa "numpy<2" "setuptools<81"

echo
echo "verifying the sidecar's imports resolve"
"$VENV/bin/python" -c "
import numpy, stark_qa
from stark_qa.evaluator import Evaluator
assert numpy.__version__ < '2', numpy.__version__
print(f'  numpy {numpy.__version__}, stark_qa.evaluator.Evaluator importable')
"
echo "done -- scoring will now use $VENV and touch no network"

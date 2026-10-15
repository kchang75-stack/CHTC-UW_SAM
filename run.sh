#!/bin/bash
set -euo pipefail

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "LS:"
ls -la

python3 --version || true
which python3 || true

echo "Done."


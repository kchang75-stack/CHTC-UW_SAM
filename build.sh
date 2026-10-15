#!/bin/bash
set -euo pipefail

echo "PWD: $PWD"
echo "Disk:"
df -h .

# Put apptainer temp/cache in the job scratch dir (big)
mkdir -p apptainer_tmp apptainer_cache
export APPTAINER_TMPDIR="$PWD/apptainer_tmp"
export APPTAINER_CACHEDIR="$PWD/apptainer_cache"

# Build
apptainer build --fakeroot sam3.sif image.def

# Verify cv2 in the built image
apptainer exec sam3.sif python3 -c "import cv2; print('cv2:', cv2.__version__)"

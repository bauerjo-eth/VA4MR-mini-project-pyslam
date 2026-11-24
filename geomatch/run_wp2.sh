#!/bin/bash
# WP2 full run: training data -> learned depth branch -> image control -> test evaluation.
# Run from the repository root with the geomatch environment active:
#     bash geomatch/run_wp2.sh
# Needs the KITTI odometry dataset in datasets/kitti_odometry/dataset (or edit BASE).
set -euo pipefail
BASE=${BASE:-datasets/kitti_odometry/dataset}
PY=${PY:-python}

if [ -z "${SKIP_COLLECT:-}" ]; then
  echo "=== $(date) collect training data (00-08)"
  $PY -m geomatch.collect_descriptor_data --base "$BASE" --sequences 00 01 02 03 04 05 06 07 08
fi

echo "=== $(date) train: depth branch"
$PY -m geomatch.train_depth_descriptor --mode depth --base "$BASE"

echo "=== $(date) train: image branch (control)"
$PY -m geomatch.train_depth_descriptor --mode image --base "$BASE"

for GAP in 10 20 80; do
  echo "=== $(date) evaluate on TEST 09-10, gap $GAP"
  $PY -m geomatch.evaluation_pipeline --config geomatch/wp2_kitti_test_config.yaml --gap $GAP
done
echo "=== $(date) ALL DONE"

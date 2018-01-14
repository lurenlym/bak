#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

DATA=./
TOOLS=/home/lzhpc/caffe/build/tools

$TOOLS/compute_image_mean $DATA/rail_train_lmdb \
  $DATA/rail_mean.binaryproto

echo "Done."

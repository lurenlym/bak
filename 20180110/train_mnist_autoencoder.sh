#!/usr/bin/env sh
set -e

~/caffe/build/tools/caffe train \
  --solver=./mnist_autoencoder_solver.prototxt $@

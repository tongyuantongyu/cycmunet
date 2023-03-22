#!/bin/bash
set -ex

mkdir build
cd build
cmake -G Ninja -DVAPOURSYNTH_PLUGIN=ON -DCMAKE_BUILD_TYPE=Release ..
ninja vs-cycmunet

mkdir -p $PREFIX/lib/vapoursynth
cp libvs-cycmunet.so $PREFIX/lib/vapoursynth/libvs-cycmunet.so

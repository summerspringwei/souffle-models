#!/bin/bash

set -xe

build_py_cmd="python3 setup.py clean ; python3 setup.py install ; pip install . ; cd .."

# Build all bindings in parallel
if [ "$1" = "p" ]; then
  cd bert && eval $build_py_cmd &
  cd efficientnet && eval $build_py_cmd &
  cd lstm && eval $build_py_cmd &
  cd mmoe && eval $build_py_cmd &
  cd swin_transformer && eval $build_py_cmd &
  wait
else # Build all bindings sequentially
  cd bert && eval $build_py_cmd
  cd efficientnet && eval $build_py_cmd
  cd lstm && eval $build_py_cmd
  cd mmoe && eval $build_py_cmd
  cd swin_transformer && eval $build_py_cmd
fi

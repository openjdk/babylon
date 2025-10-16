#!/bin/bash

# Check if the search path is provided
if [ -z "$1" ]; then
    echo "Please provide the path to search for libonnxruntime.*.dylib"
    exit 1
fi

# Search for libonnxruntime.dylib in the given path
LIB_PATH="$(find "$1" -name libonnxruntime.*.dylib -print -quit)"

if [ -z "$LIB_PATH" ]; then
    echo "libonnxruntime.*.dylib not found in $1"
    exit 1
fi

# Infer other paths based on the location of libonnxruntime.*.dylib
ONNXRT_DIR=$(dirname $(dirname $(dirname $(dirname $LIB_PATH))))
INCLUDE_DIR=$ONNXRT_DIR/include/onnxruntime
OUTPUT_DIR=src/main/java

# Run jextract
jextract --target-package oracle.code.onnx.coreml.foreign \
  -l :$LIB_PATH \
  --use-system-load-library \
  -I $INCLUDE_DIR/core/session \
  @symbols \
  --output $OUTPUT_DIR \
  $INCLUDE_DIR/core/providers/coreml/coreml_provider_factory.h
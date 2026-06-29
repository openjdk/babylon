#!/bin/bash

# Provide paths based on the location of libonnxruntime.*.[dylib|so|dll]
ONNX_INCLUDE_DIR=/path/to/onnx/include/onnxruntime/include
ONNX_GEN_AI_INCLUDE_DIR=/path/to/onnx/include/onnxruntime-genai/include
OUTPUT_DIR=../src/main/java

## Run jextract for ONNX runtime
jextract --target-package oracle.code.onnx.foreign \
  -I $ONNX_INCLUDE_DIR \
  @symbols\
  --output $OUTPUT_DIR \
  --header-class-name onnxruntime_c_api_h \
  $ONNX_INCLUDE_DIR/onnxruntime_c_api.h \
  $ONNX_INCLUDE_DIR/coreml_provider_factory.h

 ## Run jextract for Gen AI runtime
 jextract --target-package oracle.code.onnx.foreign \
   --output $OUTPUT_DIR \
   --header-class-name OrtGenApi \
   @symbolsai \
   $ONNX_GEN_AI_INCLUDE_DIR/ort_genai_c.h
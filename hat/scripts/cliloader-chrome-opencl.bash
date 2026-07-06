#!/bin/bash
cliloader -d -h \
  --chrome-call-logging \
  --chrome-device-timeline \
  --chrome-kernel-timeline \
  --chrome-device-stages \
  java @.ffi-opencl-example $*

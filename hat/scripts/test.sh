#!/bin/bash

# Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
# DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
#
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.  Oracle designates this
# particular file as subject to the "Classpath" exception as provided
# by Oracle in the LICENSE file that accompanied this code.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details (a copy is included in the LICENSE file that
# accompanied this code).
#
# You should have received a copy of the GNU General Public License version
# 2 along with this work; if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
# or visit www.oracle.com if you need additional information or have any
# questions.

if [ "$#" -lt 3 ]; then
  echo "Usage: test.sh <branch> <path> <backend...>" >&2
  exit 1
fi

GREEN="\033[0;32m"
NC="\033[0m" # No Color (reset)

## Run nvidia-smi if exists
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi
	## NVIDIA Env Variables
	export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64
	export PATH=/usr/local/cuda/bin/:$PATH
fi

branch=$1
remote_path=$2
shift 2
backends=("$@")

cd $remote_path/hat/
source setup.sh > /dev/null

# HAT Env variables
export HAT=CHECK_SSA_LOWERING

# run the test suite per backend
for backend in "${backends[@]}"
do
	echo -e "${GREEN}[running] java @.test-suite "$backend" ${NC}"
	java @.test-suite "$backend" > "$backend".txt 2> "$backend"Errors.txt
done

# Print logs
for backend in "${backends[@]}"
do
	cat "$backend".txt
done

## Run violajones
for backend in "${backends[@]}"
do
	echo -e "${GREEN}[running] java @."$backend"-example -Dheadless=true violajones.Main${NC}"
	java @."$backend"-example -Dheadless=true violajones.Main > "$backend"Violajones.log
done

for backend in "${backends[@]}"
do
	if grep -q "336faces found" "$backend"Violajones.log; then
		echo "✅ ViolaJones for backend: $backend ....... [pass]"
	else
		echo "❌ ViolaJones for backend: $backend ....... [fail]"
	fi
done

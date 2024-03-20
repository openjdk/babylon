#!/bin/bash
#
# Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
#

# copy selected cr packages into jdk.compiler

base=java/lang/reflect/code
packages="
  $base \
  $base/writer \
  $base/op \
  $base/type \
  $base/type/impl \
  "

removeclasses=""

java_base_dir=$1
jdk_compiler_dir=$2

for p in $packages; do
  mkdir -p $jdk_compiler_dir/jdk/internal/$p
  cp -r $java_base_dir/$p/*.java $jdk_compiler_dir/jdk/internal/$p/.
done

for f in $removeclasses; do
  rm $jdk_compiler_dir/jdk/internal/$f
done

find $jdk_compiler_dir/jdk/internal/$base -name "*.java" -print \
  | xargs sed -i'.bck' \
  -e 's/java\.lang\.reflect\.code/jdk\.internal\.java\.lang\.reflect\.code/g' \
  -e 's/^\/\*__\(.*\)__\*\/.*$/\1/'

find $jdk_compiler_dir/jdk/internal/$base -name "*.bck" -exec rm {} \;


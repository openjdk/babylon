#!/bin.bash
cat >/dev/null<<LICENSE
/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
LICENSE

OS=$(uname -s )
if [[ "$OS" == Linux ]]; then
  export ostype=linux
elif  [[ "$OS" == Darwin ]]; then
  export ostype=macosx
else
  echo "could not determine ostype uname -s returned ${OS}"
  exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" == x86_64 ]]; then
  export archtype=${ARCH}
elif  [[ "$ARCH" == arm64 ]]; then
  export archtype=aarch64
else
  echo "could not determine aarchtype uname -m returned ${ARCH}"
  exit 1
fi

export JAVA_HOME=${PWD}/../build/${ostype}-${archtype}-server-release/jdk
echo "exporting JAVA_HOME=${JAVA_HOME}"
if echo ${PATH} | grep ${JAVA_HOME} >/dev/null ;then
   echo 'path already contains ${JAVA_HOME}/bin'
else
   export SAFE_PATH=${PATH}
   echo 'adding ${JAVA_HOME}/bin prefix to PATH,  SAFE_PATH contains previous value'
   export PATH=${JAVA_HOME}/bin:${PATH}
fi

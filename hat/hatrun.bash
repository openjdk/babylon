#!/bin/bash 
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



if [ $# -eq 0 ]; then
   echo 'usage:'
   echo '   bash hatrun.bash [headless] backend package args ...'
   echo '       headless : Optional passes -Dheadless=true to app'
   echo '       backend  : native-opencl|native-cuda|native-hip|native-spirv|native-ptx|native-mock|java-mt|java-seq'
   echo '       package  : the examples package (and dirname under hat/examples)'
   echo '       Class name is assumed to be package.Main '
elif [[ -d build ]] ; then
   export OPTS="" 
   export VMOPTS=""
   export JARS="" 

   export VMOPTS="${VMOPTS} --add-modules jdk.incubator.code"
   export VMOPTS="${VMOPTS} --enable-preview"
   export VMOPTS="${VMOPTS} --enable-native-access=ALL-UNNAMED"
   export VMOPTS="${VMOPTS} --add-exports=java.base/jdk.internal=ALL-UNNAMED"

   export HEADLESS="${1}"
   if [[ "${HEADLESS}" = "headless" ]] ; then
      echo HEADLESS=${HEADLESS}
      shift 1  
      export OPTS="${OPTS} -Dheadless=true"
   else 
      echo "Not headless"
   fi

   export BACKEND="${1}"
   echo BACKEND=${BACKEND}
   export BACKEND_JAR=build/hat-backend-${BACKEND}-1.0.jar

   export JARS=build/hat-1.0.jar
   echo BACKEND_JAR=${BACKEND_JAR}
   if [[ ! -f ${BACKEND_JAR} ]] ;then
      echo "no backend ${BACKEND_JAR}"
      exit 1
   fi
   export JARS=${JARS}:${BACKEND_JAR}
   if [[ "$1" = "spirv" ]] ;then 
      export JARS=${JARS}:build/levelzero.jar:build/beehive-spirv-lib-0.0.4.jar;
   fi
   export OPTS="${OPTS} -Djava.library.path=build:/usr/local/lib"
   shift 1

   export EXAMPLE="${1}"
   echo EXAMPLE=${EXAMPLE}
   export EXAMPLE_JAR=build/hat-example-${EXAMPLE}-1.0.jar
   if [[  -f ${EXAMPLE_JAR} ]] ;then
      export JARS=${JARS}:${EXAMPLE_JAR}
      shift 1
   else
      echo "no example ${EXAMPLE_JAR}"
      exit 1
   fi  
   echo JARS=${JARS}
   echo VMOPTS=${VMOPTS}
   echo OPTS=${OPTS}
   echo java \${VMOPTS} \${OPTS} --class-path \${JARS} \${EXAMPLE}.Main $*
   echo java ${VMOPTS} ${OPTS} --class-path ${JARS} ${EXAMPLE}.Main $*
   java ${VMOPTS} ${OPTS} --class-path ${JARS} ${EXAMPLE}.Main $*
else
   echo No build dir
   exit 1
fi

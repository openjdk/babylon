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

function example(){
   echo $*
   headless=$1
   backend=$2
   example=$3
   shift 3
   if test "${backend}" =  "java"; then
       backend_jar=backends/shared/src/main/resources
   else
       backend_jar=maven-build/hat-backend-${backend}-1.0.jar
   fi
   echo checking backend_jar = ${backend_jar}
   if test -f ${backend_jar} -o -d ${backend_jar} ;then
      example_jar=maven-build/hat-example-${example}-1.0.jar
      echo checking example_jar = ${example_jar}
      if test -f ${example_jar} ; then
         ${JAVA_HOME}/bin/java \
            --enable-preview --enable-native-access=ALL-UNNAMED \
            --class-path maven-build/hat-1.0.jar:${example_jar}:${backend_jar} \
            --add-exports=java.base/jdk.internal=ALL-UNNAMED \
            -Djava.library.path=maven-build\
            -Dheadless=${headless} \
            ${example}.Main
      else
         echo no such example example_jar = ${example_jar}
      fi
   else
      echo no such backend backend_jar = ${backend_jar}
   fi
}

if [ $# -eq 0 ]; then 
   echo 'usage:'
   echo '   bash hatrun.bash [headless] backend package args ...'
   echo '       headless : Optional passes -Dheadless=true to app'
   echo '       package  : the examples package (and dirname under hat/examples)'
   echo '       backend  : opencl|cuda|spirv|ptx|mock'
   echo '       Class name is assumed to be package.Main '
else
   if [ $1 == headless ]; then 
      echo headless!
      shift 1
      example true $*
   else
      example false $*
   fi
fi


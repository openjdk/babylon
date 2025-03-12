cat >/dev/null<<END_OF_LICENSE
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
END_OF_LICENSE

# First lets check if this script was sourced into a bash compatible shell

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then 
   # We just bail if it was not sourced.. We want to set AND export PATH and JAVA_HOME...
   # So we must be sourced.  Otherwise we just fork a new shell, set the vars in the temp shell and exit
   echo "You must source this file ..."
   echo "Using either "
   echo "    . ${0}"
   echo "or"
   echo "    source ${0}"
   exit 1;  # exiting here is ONLY ok because we were not sourced.
else

  # We were sourced so must not exit below or we will trash the users shell ;) possibly logging them out 
  # We first need to determine the arch an os types so we can construct/test against something like
  #    macosx-aarch64-server-release
  #       ^     ^
  #       |     + ${archtype}
  #       + ${ostype}

  OS=$(uname -s )
  if [[ "$OS" == Linux ]]; then
    export ostype=linux
  elif  [[ "$OS" == Darwin ]]; then
    export ostype=macosx
  else
    # Windows?
    echo "Could not determine ostype uname -s returned ${OS}"
  fi

  ARCH=$(uname -m)
  if [[ "$ARCH" == x86_64 ]]; then
    export archtype=${ARCH}
  elif  [[ "$ARCH" ==  aarch64 ]]; then
    export archtype=aarch64
  elif  [[ "$ARCH" == arm64 ]]; then
    export archtype=aarch64
  else
    #MIPS?
    echo "Could not determine aarchtype uname -m returned ${ARCH}"
  fi

  if [[ -z "${archtype}" || -z "${ostype}" ]]; then 
     echo "Can't determine archtype and/or ostype"
  else
    # We expect either 
    #   The user provided a value for BABYLON_JDK_HOME
    # or
    #   We can locate one because we are a subdir of BABYLON using ${PWD}/..

    if [[ -z "${BABYLON_JDK_HOME}" ]]; then 
       echo "No user provided BABYLON_JDK_HOME var, we will try \${PWD}/.. = $(realpath ${PWD}/..)"
       export BABYLON_JDK_HOME=$(realpath ${PWD}/..)
       echo "We found a babylon build here ${BABYLON_JDK_HOME}"
    else
       echo "Using user supplied BABYLON_JDK_HOME ${BABYLON_JDK_HOME}"
    fi

    if command -v jextract; then 
       echo 'jextract in your PATH'
    else
       if [[ -z "${JEXTRACT_HOME}" ]]; then 
          echo "No user provided JEXTRACT_HOME var, we will try ~/jextract-22"
          if [[ -d ~/jextract-22/bin ]]; then 
             export JEXTRACT_HOME=$(realpath ~/jextract-22)
             echo "We found jextract here ${JEXTRACT_HOME}"
          fi
       else
          echo "Using user supplied JEXTRACT_HOME ${JEXTRACT_HOME}"
       fi
    fi

    if [[ -d "${JEXTRACT_HOME}/bin" ]]; then
      if echo ${PATH} | grep ${JEXTRACT_HOME} >/dev/null ;then
         echo "PATH already contains \${JEXTRACT_HOME}/bin"
      else
         export SAFE_PATH=${PATH}
         echo "Adding \${JEXTRACT_HOME}/bin prefix to PATH, SAFE_PATH contains previous value"
         export PATH=${JEXTRACT_HOME}/bin:${PATH}
      fi
    else
      echo 'You will need to add jextract to your PATH to be able to build'
      echo 'Either add it, or JEXTRACT_HOME'
    fi

    if [[ -d "${BABYLON_JDK_HOME}/build" ]]; then
      export JAVA_HOME=${BABYLON_JDK_HOME}/build/${ostype}-${archtype}-server-release/jdk
      if echo ${PATH} | grep ${JAVA_HOME} >/dev/null ;then
         echo "PATH already contains \${JAVA_HOME}/bin"
      else
         export SAFE_PATH=${PATH}
         echo "Adding \${JAVA_HOME}/bin prefix to PATH, SAFE_PATH contains previous value"
         export PATH=${JAVA_HOME}/bin:${PATH}
      fi
    else
      echo "We expected either:-"
      echo "    \${PWD} to be in a hat subdir of a compiled babylon jdk build" 
      echo "or" 
      echo "    BABYLON_JDK_HOME to be set, to a compiled babylon jdk build"
      echo ""
      echo "If you are in a hat subdir make sure babylon jdk is built ;)"
      echo ""
      echo "If you are in another dir try  "
      echo "    BABYLON_JDK_HOME=<<YOUR_PREBULT_BABYLON>> . ${0}"
      echo "or"
      echo "    BABYLON_JDK_HOME=<<YOUR_PREBULT_BABYLON>> source ${0}"
    fi
  fi
fi


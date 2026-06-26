#!/bin/bash

# Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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

##############################################
# How to use it?
# 1. Run this script with --generate-config-file  
# 2. Fill the template file: remoteTesting.conf
# 3. Run again this script without any parameters
# ############################################

display_help() {
  echo "Usage: $(basename "$0") [options]"
  echo
  echo "When running without any options, it reads the \"remoteTesting.conf\" and runs the testing framework on the remote machines."
  echo
  echo "Options:"
  echo "  --help                  Display this help message and exit."
  echo "  --generate-config-file  Generate a default configuration file and exit."
  echo "  --build-babylon         Build Babylon and HAT for all remote servers."
  echo "  --build-hat             Build HAT for all remote servers and run the tests."
  echo "  --tests                 Run HAT unittests. It assumes a previous build was performed."
  
  echo
  echo "How to use it?"
  echo "   1. Run this script with --generate-config-file "
  echo "   2. Fill the template file: remoteTesting.conf"
  echo "   3. If needed, you can run with --build-babylon to clone babylon and build HAT"
  echo "   4. (Optional if we run step 3): --build-hat assumes Babylon JDK exists and just checks for latest changes "
  echo "   5. Run tests: --tests"
  echo " "
  echo "Note: if run with no options, it will use --build-hat + --tests"
  exit 0
}

GREEN="\033[0;32m"
NC="\033[0m" # No Color (reset)

generate_config_file() {
    cat << EOF > remoteTesting.conf 
# HAT Remote Testing Settings
SERVERS=server1 server2 ...
REMOTE_USERS=user1 user2 ...

## List of Backends to test
# To test one backend
#BACKENDS=ffi-opencl
# We can also test multiple backends
BACKENDS=ffi-cuda ffi-opencl
# Specify the Babylon fork to test
FORK=https://github.com/openjdk/babylon

## Remote path. It assumes all servers use the same path
REMOTE_PATH=repos/babylon/hat
## Branch to test
BRANCH=code-reflection
EOF
  echo "✅ Default configuration file 'remoteTesting.conf' has been generated."
  exit 0
}

read_config_file() {
  CONFIG_FILE="remoteTesting.conf"

  # Check if the config file exists
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: $CONFIG_FILE not found."
    echo "Run this script with --generate-config-file to generate a template."
    exit 1
  fi

  ## Process the config file
  while IFS='=' read -r key value
  do
    if [[ -z "$key" || "$key" =~ ^# ]]; then
      continue
    fi

    case "$key" in
      "SERVERS") SERVERS="$value" ;;
      "REMOTE_USERS") REMOTE_USERS="$value" ;;
      "BACKENDS") BACKENDS=($value) ;;
      "REMOTE_PATH") REMOTE_PATH="$value" ;;
      "BRANCH") BRANCH="$value" ;;
      "FORK") FORK="$value" ;;
    esac
  done < "$CONFIG_FILE"

  isSet=1
  if [[ -z $SERVERS ]]; then
    echo "❌ SERVERS is not set."
    isSet=0
  fi
  if [[ -z $REMOTE_USERS ]]; then
    echo "❌ REMOTE_USERS is not set."
    isSet=0
  fi
  if [[ -z $BACKENDS ]]; then
    echo "❌ BACKENDS is not set."
    isSet=0
  fi
  if [[ -z $REMOTE_PATH ]]; then
    echo "❌ REMOTE_PATH is not set."
    isSet=0
  fi
  if [[ -z $BRANCH ]]; then
    echo "❌ BRANCH is not set."
    isSet=0
  fi
  if [[ -z $FORK ]]; then
    echo "❌ FORK is not set."
    isSet=0
  fi

  if [[ "$isSet" -eq 0 ]]; then
	  exit
  fi

  echo
  echo "Servers    : $SERVERS"
  echo "Users      : $REMOTE_USERS"
  echo "Backends   : ${BACKENDS[@]}"
  echo "Remote Path: $REMOTE_PATH"
  echo "Fork       : $FORK"
  echo "Branch     : $BRANCH"
  echo

  read -ra listOfServers <<< $SERVERS
  read -ra listOfUsers <<< $REMOTE_USERS
}

build_babylon() {
  echo "Build Babylon and HAT" 
  read_config_file
  for index in "${!listOfServers[@]}"
  do
    server=${listOfServers[$index]}
    user=${listOfUsers[$index]}
    echo -e "\n${GREEN}[info] ssh -t $user@$server 'bash -s -- $FORK $BRANCH $REMOTE_PATH' < scripts/build_babylon.sh ${NC}"
    ssh -t $user@$server "bash -s -- ${FORK} ${BRANCH} ${REMOTE_PATH}" < scripts/build_babylon.sh
  done
}

build_hat() {
  read_config_file
  for index in "${!listOfServers[@]}"
  do
    server=${listOfServers[$index]}
    user=${listOfUsers[$index]}
    list_backends=${BACKENDS[@]}
    echo -e "\n${GREEN}[info] ssh -t $user@$server 'bash -s -- $BRANCH $REMOTE_PATH ${BACKENDS[@]}' < scripts/compile.sh ${NC}"
    ssh -t $user@$server "bash -s -- ${BRANCH} ${REMOTE_PATH} ${list_backends}" < scripts/compile.sh
  done
}

run_tests_hat() {
  read_config_file
  for index in "${!listOfServers[@]}"
  do
    server=${listOfServers[$index]}
    user=${listOfUsers[$index]}
    list_backends=${BACKENDS[@]}
    echo -e "\n${GREEN}[info] ssh -t $user@$server 'bash -s -- $BRANCH $REMOTE_PATH ${BACKENDS[@]}' < scripts/test.sh ${NC}"
    ssh -t $user@$server "bash -s -- ${BRANCH} ${REMOTE_PATH} ${list_backends}" < scripts/test.sh
  done
}

build_and_and_test() {
  read_config_file
  for index in "${!listOfServers[@]}"
  do
    server=${listOfServers[$index]}
    user=${listOfUsers[$index]}
    list_backends=${BACKENDS[@]}
    echo -e "\n${GREEN}[info] ssh -t $user@$server 'bash -s -- $BRANCH $REMOTE_PATH ${BACKENDS[@]}' < scripts/compile.sh ${NC}"
    ssh -t $user@$server "bash -s -- ${BRANCH} ${REMOTE_PATH} ${list_backends}" < scripts/compile.sh
    echo -e "\n${GREEN}[info] ssh -t $user@$server 'bash -s -- $BRANCH $REMOTE_PATH ${BACKENDS[@]}' < scripts/test.sh ${NC}"
    ssh -t $user@$server "bash -s -- ${BRANCH} ${REMOTE_PATH} ${list_backends}" < scripts/test.sh
  done
}

main() {
  while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
      --help)
        display_help
        exit
        ;;
      --generate-config-file)
        generate_config_file
        exit 
        ;;
      --build-babylon)
        build_babylon
        exit 0
        ;;
      --build-hat)
        build_hat
        exit 0
        ;;
      --tests)
        run_tests_hat
        exit 0
        ;;
      *)
        # Unknown option
        echo "Error: Unknown option '$key'"
        echo "Use --help for a list of available options."
        exit 1
        ;;
    esac
  done
  ## No options, builds HAT and run the tests
  build_and_and_test
}

main $@



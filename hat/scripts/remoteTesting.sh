#!/bin/env/bash

# Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
  echo
  echo "How to use it?"
  echo "   1. Run this script with --generate-config-file "
  echo "   2. Fill the template file: remoteTesting.conf"
  echo "   3. Run again this script without any parameters "
  exit 0
}

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

## Remote path. It assumes all servers use the same path
REMOTE_PATH=repos/babylon/hat
## Branch to test
BRANCH=code-reflection
EOF
  echo "✅ Default configuration file 'remoteTesting.conf' has been generated."
  exit 0
}

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --help)
      display_help
      ;;
    --generate-config-file)
      generate_config_file
      ;;
    *)
      # Unknown option
      echo "Error: Unknown option '$key'"
      echo "Use --help for a list of available options."
      exit 1
      ;;
  esac
done

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
    "BACKENDS") BACKENDS=$value ;;
    "REMOTE_PATH") REMOTE_PATH="$value" ;;
    "BRANCH") BRANCH="$value" ;;
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

if [[ "$isSet" -eq 0 ]]; then
	exit
fi

echo
echo "Servers    : $SERVERS"
echo "Users      : $REMOTE_USERS"
echo "Backends   : $BACKENDS"
echo "Remote Path: $REMOTE_PATH"
echo "Branch     : $BRANCH"
echo

read -ra listOfServers <<< $SERVERS
read -ra listOfUsers <<< $REMOTE_USERS

for backend in $BACKENDS
do
echo $backend > /dev/null
done


for index in "${!listOfServers[@]}"
do

server=${listOfServers[$index]}
user=${listOfUsers[$index]}

echo "ssh $user@$server"
ssh $user@$server << EOF
cd "$REMOTE_PATH"
git checkout $BRANCH
git pull

# compile
source setup.sh
java @hat/clean 
java @hat/bld > compilation.log

# run the test suite per backend
for backend in $BACKENDS
do
echo "[running] java @hat/test suite $backend  > $backend.txt"
java @hat/test suite $backend  > $backend.txt
done

# Print logs
for backend in $BACKENDS
do
cat $backend.txt
done
EOF
done

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
  echo "Usage: test.sh <fork> <branch> <path>" >&2
  exit 1
fi

GREEN="\033[0;32m"
NC="\033[0m" # No Color (reset)

fork=$1
branch=$2
remote_path=$3

echo $fork
echo $branch
echo $remote_path

if [ ! -d $remote_path ]; 
then 
  mkdir -p $remote_path
  cd $$remote_path
  git clone $fork $$remote_path
fi

#Assuming the remote path ends with babylon
cd $remote_path
git fetch --all
git checkout $branch
git pull

echo "bash configure --with-boot-jdk=$HOME/.sdkman/candidates/java/current"
bash configure --with-boot-jdk="$HOME/.sdkman/candidates/java/current" > jvmconfig.log
make clean
make images 

## Build HAT 
cd hat 
if [ ! -d jextract-25 ];
then
  echo "ARCHITECTIRE $(uname -m)"
  if [[ "$(uname -m)" == "x86_64" ]]; then
      wget https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_linux-x64_bin.tar.gz
      tar xvzf openjdk-25-jextract+2-4_linux-x64_bin.tar.gz 
  elif [[ "$(uname -m)" == "arm64" ]]; then
      wget https://download.java.net/java/early_access/jextract/25/2/openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz
      tar xvzf openjdk-25-jextract+2-4_macos-aarch64_bin.tar.gz 
  fi
  echo "export PATH=$PWD/jextract-25/bin:$PATH" >> setup.sh
  echo "source env.bash" >> setup.sh
fi

source setup.sh > /dev/null 2> /dev/null
mvn clean package 

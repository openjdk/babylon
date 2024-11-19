#!/bin/zsh

javac --enable-preview --source 24 -d . test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java

for fp in `find build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect -name "*.class"`; do \
  java --enable-preview -ea -cp .:$(dirname $fp) OpFieldToMethodBuilder $fp \
  ; done

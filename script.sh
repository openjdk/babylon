#!/bin/bash

javac --enable-preview --source 24 -d . test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java
java --enable-preview -ea OpFieldToMethodBuilder \
  $(find build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect -name "*.class")
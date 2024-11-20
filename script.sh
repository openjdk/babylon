#!/bin/zsh

javac --enable-preview --source 24 -d . test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java
touch a.csv
echo "writing data to a.csv ..."
echo "classfile, original_size, new_size" > a.csv
for fp in `find build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect -name "*.class"`; do \
  java --enable-preview -ea -cp .:$(dirname $fp) OpFieldToMethodBuilder $fp | sed 's/[[:space:]]/,/g' | sed "s/^[^,]*/$(basename $fp)/" >> a.csv
  ; done

awk '!seen[$0]++' a.csv > tmp && mv tmp a.csv

awk -F ',' '$2 < $3 {print $0}' a.csv > tmp && mv tmp a.csv

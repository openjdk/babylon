#!/bin/zsh

babylon_dir=$PWD
target_file="$babylon_dir/b.csv"
echo "" > $target_file
echo "writing data to $target_file ..."
for fp in `find hat/build/hat-example*classes -name "*.class"`; do \
  fdir=$(dirname $fp)
  cp test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java $fdir
  cd $fdir
  pkgName=$(dirname $(sed 's#^.*classes/\(.*\)#\1#g'<<< $fp) | sed 's#/#.#g')
  echo "package $pkgName;" > tmp && cat OpFieldToMethodBuilder.java >> tmp && mv tmp OpFieldToMethodBuilder.java
  # we want example_name/../class_file
  example_name=$(sed 's#^.*classes/\(.*\)#\1#g'<<< $fp | sed 's#/#.#g')
  java --enable-preview -ea OpFieldToMethodBuilder.java $(basename $fp) | awk "{\$1=\"$example_name\"; print}" OFS=, >> $target_file
  rm OpFieldToMethodBuilder.java
  cd $babylon_dir
; done
awk '!seen[$0]++' $target_file > tmp && mv tmp $target_file
awk -F ',' '$2 < $3' $target_file > tmp && mv tmp $target_file
echo "classfile, original_size, new_size" > tmp && cat $target_file >> tmp && mv tmp $target_file

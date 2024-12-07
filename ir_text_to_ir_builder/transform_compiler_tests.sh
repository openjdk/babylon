#!/bin/zsh

babylon_dir=~/babylon
java_program=$babylon_dir/test/jdk/java/lang/reflect/code/writer/IRTextToIRBuilder.java
target_dir=ir_text_to_ir_builder
compiler_tests_dir=$babylon_dir/build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect
for cfp in `find $compiler_tests_dir -name "*Test.class"`; do \
  cfp_from_project_root=$(sed "s#$babylon_dir/##" <<< $cfp)
  mkdir -p $(dirname $cfp_from_project_root)
  java --enable-preview -cp $(dirname $cfp) $java_program $cfp > $cfp_from_project_root
  ; done

echo "Transformed classfiles are in $(dirname $0)/$(sed "s#$babylon_dir/##" <<< $compiler_tests_dir)"

# remember that compiler tests contains the expected text IR, so the new cf will still contains those in its CP

# build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect/TryTest.d/TryTest.class
## /Users/mabbay/babylon/test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java
#javac --enable-preview --source 24 -d . test/jdk/java/lang/reflect/code/writer/OpFieldToMethodBuilder.java
#echo "writing data to a.csv ..."
#cho "" > a.csv
#for fp in `find build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect -name "*.class"`; do \
 # java --enable-preview -ea -cp .:$(dirname $fp) OpFieldToMethodBuilder $fp | sed 's/[[:space:]]/,/g' | sed "s/^[^,]*/$(basename $fp)/" >> a.csv
  #; done

#ak '!seen[$0]++' a.csv > tmp && mv tmp a.csv

#awk -F ',' '$2 < $3' a.csv > tmp && mv tmp a.csv

#echo "classfile, original_size, new_size" > tmp && cat a.csv >> tmp && mv tmp a.csv

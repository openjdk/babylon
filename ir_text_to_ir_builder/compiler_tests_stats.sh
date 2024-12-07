#!/bin/zsh

target_file=compiler_tests_stats.csv
echo "classfile, original_size, new_size" > $target_file

target_dir=ir_text_to_ir_builder
for cfp in `find ../build/macosx-aarch64-server-release/JTwork/classes/tools/javac/reflect -name "*Test.class"`; do \
  original_size=$(awk '{print $1}' <<< $(wc -c $cfp))
  path_to_new_cf=$(cut -d'/' -f2- <<< $cfp)
  new_size=$(awk '{print $1}' <<< $(wc -c $path_to_new_cf))
  echo "$(basename $cfp) $original_size $new_size" >> $target_file
  ; done

echo "Statistics are in file: $(dirname $0)/$target_file"
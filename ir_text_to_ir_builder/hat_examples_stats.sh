#!/bin/zsh

target_file=hat_examples_stats.csv
echo "classfile, original_size, new_size" > $target_file

target_dir=ir_text_to_ir_builder
babylon_dir=~/babylon

for cfp in `find $babylon_dir/hat/build/hat-example*classes -name "*.class"`; do \
  original_size=$(awk '{print $1}' <<< $(wc -c $cfp))
  path_to_new_cf=$(sed "s#$babylon_dir/##" <<< $cfp)
  new_size=$(awk '{print $1}' <<< $(wc -c $path_to_new_cf))
  if [ $new_size -gt $original_size ]; then
    #
    hat_example_name=$(sed 's#^.*classes/\(.*\)#\1#g'<<< $cfp | sed 's#/#.#g')
    echo "$hat_example_name $original_size $new_size" >> $target_file
  fi
  ; done

echo "Statistics are in file: $(dirname $0)/$target_file"
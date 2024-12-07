#!/bin/zsh

babylon_dir=~/babylon
java_program=$babylon_dir/test/jdk/java/lang/reflect/code/writer/IRTextToIRBuilder.java
for cfp in `find $babylon_dir/hat/build/hat-example*classes -name "*.class"`; do \
  cfp_from_project_root=$(sed "s#$babylon_dir/##" <<< $cfp)
  mkdir -p $(dirname $cfp_from_project_root)
  java --enable-preview -ea $java_program $cfp > $cfp_from_project_root
; done
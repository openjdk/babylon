#!/bin/bash

$JEXTRACT_DIR/bin/jextract --output src/gen/java -I /usr/include -t oneapi.levelzero level-zero/include/ze_api.h
$JAVA_HOME/bin/javac -cp target/classes -d target/classes src/gen/java/oneapi/levelzero/*.java
$JAVA_HOME/bin/jar cf levelzero.jar -C target/classes/ .
mvn install:install-file -Dfile=levelzero.jar -DgroupId=oneapi -DartifactId=levelzero -Dversion=0.0.1 -Dpackaging=jar
cp levelzero.jar ../../../maven-build/

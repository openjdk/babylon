echo $JAVA_HOME

RELEASE=$(java -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p')

ONNXRUNTIME_JAR=~/.m2/repository/com/microsoft/onnxruntime/onnxruntime/1.20.0/onnxruntime-1.20.0.jar

# Compile the classes
javac --release $RELEASE \
  -d target/classes \
  -cp  $ONNXRUNTIME_JAR \
  src/main/java/oracle/code/onnx/OnnxNumber.java \
  src/main/java/oracle/code/onnx/foreign/*.java \
  src/main/java/oracle/code/onnx/coreml/foreign/*.java \
  src/main/java/oracle/code/onnx/coreml/*.java

javac --release $RELEASE \
  -d target/test-classes \
  -cp target/classes:$ONNXRUNTIME_JAR \
  src/test/java/oracle/code/onnx/fer/*.java

cp -r src/test/resources/oracle/code/onnx/fer/ target/test-classes/oracle/code/onnx/fer/

# Run the program
java --enable-preview \
  -Djava.library.path=$ONNXRUNTIME_JAR!/ai/onnxruntime/native/osx-aarch64/libonnxruntime.dylib \
  -cp target/test-classes:target/classes:$ONNXRUNTIME_JAR \
  oracle.code.onnx.fer.FERCoreMLDemo
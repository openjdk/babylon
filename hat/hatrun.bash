function example(){
  backend=$1
  example=$2
  example_class=$3
  if test "${backend}" = "java"; then
    backend_jar=backends/shared/src/main/resources
  else
    backend_jar=backends/${backend}/target/hat-backend-${backend}-1.0.jar
  fi

  echo checking backend_jar = ${backend_jar}
  if ! test -e ${backend_jar} ;then
    echo no such backend backend_jar = ${backend_jar}
    exit 1
  fi

  backend_native_path=build/backends/${backend}
  if ! test "${backend}" = "java"; then
    echo checking backend_native_path = ${backend_native_path}
    if ! test -d ${backend_native_path} ;then
      echo no such native path backend_native_path = ${backend_native_path}
      exit 1
    fi
  fi

  example_jar=examples/${example}/target/hat-example-${example}-1.0.jar
  echo checking example_jar = ${example_jar}
  if ! test -f ${example_jar} ;then
    echo no such example example_jar = ${example_jar}
    exit 1
  fi

  ${JAVA_HOME}/bin/java \
    --enable-preview --enable-native-access=ALL-UNNAMED \
    --class-path hat/target/hat-1.0.jar:${example_jar}:${backend_jar} \
    --add-exports=java.base/jdk.internal=ALL-UNNAMED \
    -Djava.library.path=${backend_native_path} \
    -Dheadless=true \
    ${example}.${example_class}
}

example $*



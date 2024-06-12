function example(){
   backend=$1
   example=$2
   example_class=$3
   if test "${backend}" =  "java"; then
       backend_jar=backends/shared/src/main/resources
   else
       backend_jar=maven-build/hat-backend-${backend}-1.0.jar
   fi
   echo checking backend_jar = ${backend_jar}
   if test -f ${backend_jar} -o -d ${backend_jar} ;then
      example_jar=maven-build/hat-example-${example}-1.0.jar
      echo checking example_jar = ${example_jar}
      if test -f ${example_jar} ; then
         ${JAVA_HOME}/bin/java \
            --enable-preview --enable-native-access=ALL-UNNAMED \
            --class-path hat/target/hat-1.0.jar:${example_jar}:${backend_jar} \
            --add-exports=java.base/jdk.internal=ALL-UNNAMED \
            -Djava.library.path=maven-build\
            -Dheadless=true \
            ${example}.${example_class}
      else
         echo no such example example_jar = ${example_jar}
      fi
   else
      echo no such backend backend_jar = ${backend_jar}
   fi
}

example $*



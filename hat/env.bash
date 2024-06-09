OS=$(uname -s )
if [[ "$OS" == Linux ]]; then
  export ostype=linux
elif  [[ "$OS" == Darwin ]]; then
  export ostype=macosx
else 
  echo "could not determine ostype uname -s returned ${OS}"
  exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" == x86_64 ]]; then
  export archtype=${ARCH}
elif  [[ "$ARCH" == arm64 ]]; then
  export archtype=aarch64
else 
  echo "could not determine aarchtype uname -m returned ${ARCH}"
  exit 1
fi

export JAVA_HOME=${PWD}/../build/${ostype}-${archtype}-server-release/jdk
echo "exporting JAVA_HOME=${JAVA_HOME}"
if echo ${PATH} | grep ${JAVA_HOME} >/dev/null ;then 
   echo 'path already contains ${JAVA_HOME}/bin'
else
   export SAFE_PATH=${PATH}
   echo 'adding ${JAVA_HOME}/bin prefix to PATH,  SAFE_PATH contains previous value'
   export PATH=${JAVA_HOME}/bin:${PATH}
fi

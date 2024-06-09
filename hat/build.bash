cmake -B build
for backend in \
   opencl\
   mock\
   ptx\
;do
    cmake --build build --target ${backend}_backend
done

mvn clean compile jar:jar

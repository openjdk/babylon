git clone https://github.com/beehive-lab/beehive-spirv-toolkit.git
cd beehive-spirv-toolkit
mvn clean install
cd ..
cp beehive-spirv-toolkit/lib/target/beehive-spirv-lib-0.0.4.jar ../../../maven-build/

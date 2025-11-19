[ -d build/job.classes ] && echo "removing old job classes dir " && rm -rf build/job.classes
echo "creating job classes dir " && mkdir -p build/job.classes
echo "compiling job classes " && javac -d build/job.classes --source-path hat/job/src/main/java $(find hat/job/src/main/java/job -name "*.java")
echo "creating hat/job.jar" && jar cf hat/job.jar -C build/job.classes job

echo spaces at end of lines
find . \
   -name "*.iml" \
   -o -name "*.bash" \
   -o -name "*.xml" \
   -o -name "*.java" \
   -o -name "*.h" \
   -o -name "*.md" \
   -o -name "*.cpp" \
   -o -name CMakeFiles.list \
   | xargs grep "  *$" \
   | cut -d: -f1 \
   | sort -u

echo tabs
find . \
   -name "*.iml" \
   -o -name "*.bash" \
   -o -name "*.xml" \
   -o -name "*.java" \
   -o -name "*.h" \
   -o -name "*.md" \
   -o -name "*.cpp" \
   -o -name CMakeFiles.list \
   | xargs grep "\t" \
   | cut -d: -f1 \
   | sort -u


find . -type f -name pom.xml -print0 | xargs -0 sed -i '' 's|<release>26|<release>27|g'

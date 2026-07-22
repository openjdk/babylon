/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
void main(String[] args) throws IOException{
      Files.walkFileTree(Paths.get("./"), new SimpleFileVisitor<>() {
         private static final Set<String> TARGET_EXTENSIONS = Set.of(".java", ".h", ".cpp", ".md");
         private static final Set<String> SKIP_COPYRIGHT_EXTENSIONS = Set.of(".md");

         private static final Set<String> TARGET_FILES = Set.of("pom.xml"/* "CMakeLists.txt"*/);
         private static final Pattern COPYRIGHT_PATTERN = Pattern.compile("^.*Copyright.*202[0-9].*(Intel|Oracle).*$", Pattern.MULTILINE);

         private final Set<String> IGNORED_DIRS = Set.of("target",  "build", "robertograham","hip");
            @Override public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) {
               return IGNORED_DIRS.contains(dir.getFileName().toString())?FileVisitResult.SKIP_SUBTREE:FileVisitResult.CONTINUE;
            }
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
               var name = file.getFileName().toString();
               if (TARGET_FILES.contains(name) || TARGET_EXTENSIONS.stream().anyMatch(name::endsWith)){
                  try {
                     var lines = Files.readAllLines(file, StandardCharsets.UTF_8);
                     var tab = new ArrayList<Integer>();
                     var eolWs = new ArrayList<Integer>();
                     var copyright = SKIP_COPYRIGHT_EXTENSIONS.stream().anyMatch(name::endsWith);
                     for (int i = 0; i < lines.size(); i++) {
                        var line = lines.get(i);
                        copyright = copyright || COPYRIGHT_PATTERN.matcher(line).find();
                        if (line.contains("\t")) {
                           tab.add(i + 1);
                        }
                        if (line.matches(".*\\s+$")) {
                           eolWs.add(i + 1);
                        }
                     }
                     if (!copyright){
                        IO.println("[NO COPYRIGHT] " + file);
                     }
                     if (!tab.isEmpty()) {
                        IO.println("[         Tab] " + file + " (" + tab + ")");
                     }
                     if (!eolWs.isEmpty()) {
                        IO.println("[      EOL WS] " + file + " (" + eolWs + ")");
                     }
                  } catch (IOException e) {
                     throw new RuntimeException("Could not read file: " + file);
                  }
               }
               return  FileVisitResult.CONTINUE;
            }
      });
}

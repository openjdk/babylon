/*
 *
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

import static java.lang.IO.println;

void main(String[] args){
  Script.DirEntry.current()
    .subDirs()
    .filter(dir -> dir.matches("^.*(hat|tools|wraps|examples|backends|docs|core|extractions)$"))
    .forEach(dir->dir
       .findFiles()
          .filter((path)->Pattern.matches("^.*/.*\\.(java|cpp|h|hpp|md)$", path.toString()))
          .filter((path)->!Pattern.matches("^.*examples/life/src/main/java/io.*$", path.toString())) // Life example has some open source files
          .filter((path)->!Pattern.matches("^.*CMakeFiles.*$", path.toString()))
          .filter((path)->!Pattern.matches("^.*extractions.*/src/main/.*$", path.toString()))
          .map(path->new Script.SearchableTextFile(path))
          .forEach(textFile ->{
             if (!textFile.hasSuffix("md")
               && !textFile.grep(Pattern.compile("^.*Copyright.*202[0-9].*(Intel|Oracle).*$"))) {
                  println("ERR NO LICENCE " + textFile.path());
             }
             textFile.lines().forEach(line -> {
               if (line.grep(Pattern.compile("^.*\\t.*"))) {
                  println("ERR        TAB " + textFile.path() + ":" + line.line() + "#" + line.num());
               }
               if (line.grep(Pattern.compile("^.* $"))) {
                  println("ERR EOL WSPACE " + textFile.path() + ":" + line.line() + "#" + line.num());
               }
            });
          })
   );
}

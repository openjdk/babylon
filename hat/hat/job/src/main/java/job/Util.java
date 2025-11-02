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
package job;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.regex.Pattern;

public class Util {


    public static boolean grep(Pattern pattern, String str){
        return pattern.matcher(str).matches();
    }

    public static boolean grepLines(Pattern pattern, List<String> lines){
        var result=new boolean[]{false};
        lines.forEach(line->{
            result[0] |= grep(pattern, line);
        });
        return result[0];
    }


    public static boolean grepLines(List<Pattern> patterns, List<String> lines){
        for (var pattern:patterns){
            if (grepLines(pattern, lines)){
                return true;
            }
        }
        return false;
    }

    public static boolean grepLines(Pattern pattern, Path path){
        try{
            return grepLines(pattern, Files.readAllLines(path));
        }catch(IOException i){
            return false;
        }
    }

    public static boolean grepLines(List<Pattern> patterns, Path path){
        try{
            return grepLines(patterns, Files.readAllLines(path));
        }catch(IOException i){
            return false;
        }
    }

    public static void recurse(Path dir, Predicate<Path> dirPredicate, Predicate<Path> filePredicate, Consumer<Path> consumer){
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (Files.isDirectory(entry)) {
                    if (dirPredicate.test(entry)) {
                        recurse(entry, dirPredicate,filePredicate, consumer);
                   // }else{
                     //   System.out.println(entry + "failed dir predicate" );
                    }
                }else if (filePredicate.test(entry)){
                    consumer.accept(entry);
                }
            }
        }catch(IOException ioe){
            throw new IllegalStateException(ioe);
        }
    }
}

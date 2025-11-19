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

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class ForkExec {
    public record Result(
            Dependency dependency,
            Path path,
            Opts opts,
            int status,
            List<String> stdErrAndOut){
    }
    static Result forkExec(Dependency dependency, Path path, Opts opts) {
        try {
            List<String> stdErrAndOut = new ArrayList<>();
            Process process = new ProcessBuilder()
                    .directory(path.toFile())
                    .command(opts.opts)
                    .redirectErrorStream(true)
                    .start();
            new BufferedReader(new InputStreamReader(process.getInputStream())).lines().forEach(line->{
                System.out.println(line);
                stdErrAndOut.add(line);
            });
            process.waitFor();
            return new Result(dependency, path, opts, process.exitValue(), stdErrAndOut);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static class Opts {
        List<String> opts = new ArrayList<>();
        private Opts(){

        }
        Opts add(String ...opts){
            this.opts.addAll(List.of(opts));
            return this;
        }

        public static Opts of(String executable) {
            Opts opts = new Opts();
            opts.add(executable);
            return opts;
        }

        @Override
        public String toString() {
            return String.join(" ", opts);
        }

    }
}

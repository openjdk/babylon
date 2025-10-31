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

import java.util.function.Consumer;

public class Reporter {
    public final Consumer<String> command = System.out::println;
    public final Consumer<String> progress = System.out::println;
    public final Consumer<String> error = System.out::println;
    public final Consumer<String> info = System.out::println;
    public final Consumer<String> warning = System.out::println;
    public final Consumer<String> note = System.out::println;

    public void command(Dependency dependency, String command) {
        if (dependency != null) {
            this.command.accept("# " + dependency.id().projectRelativeHyphenatedName() + " command line ");
        }
        this.command.accept(command);
    }

    public void progress(Dependency dependency, String command) {
        if (dependency != null) {
            progress.accept("# " + dependency.id().projectRelativeHyphenatedName() + " " + command);
        }
    }

    public void error(Dependency dependency, String command) {
        if (dependency != null) {
            error.accept("# " + dependency.id().projectRelativeHyphenatedName() + " error ");
        }
        error.accept(command);
    }

    public void info(Dependency dependency, String command) {
        // if (dependency != null) {
        //     info.accept("# "+dependency.id().projectRelativeHyphenatedName+" info ");
        //  }
        info.accept(command);
    }

    public void note(Dependency dependency, String command) {
        //  if (dependency != null) {
        //    note.accept("# "+dependency.id().projectRelativeHyphenatedName+" note ");
        //  }
        note.accept(command);
    }

    public void warning(Dependency dependency, String command) {
        //   if (dependency != null) {
        //      warning.accept("# "+dependency.id().projectRelativeHyphenatedName+" warning ");
        //  }
        warning.accept(command);
    }

    static Reporter verbose = new Reporter();
    public static Reporter commandsAndErrors = new Reporter() {
        @Override
        public void warning(Dependency dependency, String command) {

        }

        @Override
        public void info(Dependency dependency, String command) {

        }

        @Override
        public void note(Dependency dependency, String command) {

        }

        @Override
        public void progress(Dependency dependency, String command) {

        }

    };

    public static Reporter progressAndErrors = new Reporter() {
        @Override
        public void warning(Dependency dependency, String command) {

        }

        @Override
        public void info(Dependency dependency, String command) {

        }

        @Override
        public void note(Dependency dependency, String command) {

        }

        @Override
        public void command(Dependency dependency, String command) {

        }

        public void progress(Dependency dependency, String command) {
            if (dependency != null) {
                progress.accept(dependency.id().projectRelativeHyphenatedName() + ":" + command);
            }
        }
    };
}

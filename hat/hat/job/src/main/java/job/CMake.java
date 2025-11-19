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

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

public class CMake extends DependencyImpl<CMake> implements Dependency.Buildable, Dependency.WithPath {
    public ForkExec.Result cmake(Consumer<String> lineConsumer, List<String> tailopts) {
        ForkExec.Opts opts = ForkExec.Opts.of("cmake");
        tailopts.forEach(opts::add);
        id.project().reporter.command(this, opts.toString());
        id.project().reporter.progress(this, opts.toString());
        var result =  ForkExec.forkExec(this, id.project().rootPath(), opts);
        result.stdErrAndOut().forEach((line) -> {
            lineConsumer.accept(line);
            id().project().reporter.info(this, line);
        });

        if (result.status()!=0){
            id().project().reporter.error(this, opts.toString());
            throw new RuntimeException("CMake failed");
        }
        return result;
    }

    @Override
    public List<Path> generatedPaths() {
        throw new IllegalStateException("who called me");
    }

    ForkExec.Result cmake(Consumer<String> lineConsumer, String... opts) {
        return cmake(lineConsumer, List.of(opts));
    }

    public ForkExec.Result cmakeInit(Consumer<String> lineConsumer) {
        return cmake(lineConsumer, "--fresh", "-DHAT_TARGET=" + id().project().buildPath(), "-B", cmakeBuildDir().toString(), "-S", cmakeSourceDir().toString());
    }

    public ForkExec.Result cmakeBuildTarget(Consumer<String> lineConsumer, String target) {
        return cmake(lineConsumer, "--build", cmakeBuildDir().toString(), "--target", target);
    }

    public ForkExec.Result cmakeBuild(Consumer<String> lineConsumer) {
        return cmake(lineConsumer, "--build", cmakeBuildDir().toString());
    }

    public ForkExec.Result cmakeClean(Consumer<String> lineConsumer) {
        return cmakeBuildTarget(lineConsumer, "clean");
    }


    @Override
    public boolean build() {
        cmakeInit(_ -> {
        });
        cmakeBuild(_ -> {
        });
        return false;
    }

    @Override
    public boolean clean() {
        cmakeInit(_ -> {
        });
        cmakeClean(_ -> {
        });
        return false;
    }

    final Path cmakeSourceDir;
    final Path cmakeBuildDir;

    Path cmakeSourceDir() {
        return cmakeSourceDir;
    }

    Path cmakeBuildDir() {
        return cmakeBuildDir;
    }

    final Path CMakeLists_txt;

    protected CMake(Project.Id gsn, Path cmakeSourceDir, Set<Dependency> dependencies) {
        super(gsn, dependencies);
        if (id.path() != null && !Files.exists(id.path())) {
            System.err.println("The path does not exist: " + id.path());
        }
        this.cmakeSourceDir = cmakeSourceDir;
        this.cmakeBuildDir = cmakeSourceDir.resolve("build");
        this.CMakeLists_txt = cmakeSourceDir.resolve("CMakeLists.txt");
    }

    protected CMake(Project.Id id, Set<Dependency> dependencies) {
        this(id, id.path(), dependencies);
    }

    public static CMake of(Project.Id id, Set<Dependency> dependencies) {
        return new CMake(id, dependencies);
    }

    public static CMake of(Project.Id id, Dependency... dependencies) {
        return of(id, Set.of(dependencies));
    }

}

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
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

public class CMake extends DependencyImpl<CMake> implements Dependency.Buildable, Dependency.WithPath {

    public ForkExec.Result cmake(Consumer<String> lineConsumer, List<String> tailopts) {
        StringList stringList = StringList.of()
                .add("cmake")
                .add(tailopts);

        if (id().project().cmakeOpts().command()) {
            System.out.println(stringList);
        }
        if (id().project().cmakeOpts().progress()) {
            System.out.println("cmake "+id().fullHyphenatedName());
        }
        if (id().project().cmakeOpts().verbose()) {
            System.out.println(stringList);
        }
        var result =  ForkExec.forkExec(this, false, id.project().rootPath(), stringList);
        result.stdErrAndOut().forEach((line) -> {
            lineConsumer.accept(line);
            if (id().project().cmakeOpts().verbose()) {
               System.out.println( line);
            }
        });

        if (result.status()!=0){
            System.err.println( stringList);
            throw new RuntimeException("CMake failed");
        }
        return result;
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
        cmakeInit(_ -> {});
        cmakeBuild(_ -> {});
        return false;
    }

    @Override
    public boolean clean(boolean verbose) {
        cmakeInit(_ -> {});
        cmakeClean(_ -> {});
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

    public  interface Config extends CommonConfig<Config>{
        boolean command();

        boolean verbose();
        boolean progress();

        boolean warnings();
        List<String> vmOpts();

        record ConfigImpl(boolean command, boolean warnings, boolean progress, boolean verbose, List<String> vmOpts) implements Config {
        }

        static Config of(boolean command, boolean warnings, boolean progress, boolean verbose, List<String> vmOpts) {
            return new ConfigImpl(command, warnings,progress,verbose, vmOpts);
        }

        interface Builder extends Config {
            Builder command(boolean f);
            Builder warnings(boolean f);
            Builder progress(boolean f);
            Builder verbose(boolean f);

            Builder vmOpt(String... s);


            class Impl implements Builder {
                boolean command;
                boolean warnings;
                boolean progress;
                boolean verbose;
                List<String> vmOpts = new ArrayList<>();

                @Override
                public Builder command(boolean f) {
                    command = f;
                    return this;
                }
                @Override
                public Builder warnings(boolean f) {
                    warnings= f;
                    return this;
                }
                @Override
                public Builder progress(boolean f) {
                    progress= f;
                    return this;
                }

                @Override
                public Builder verbose(boolean f) {
                    verbose = f;
                    return this;
                }

                @Override
                public Builder vmOpt(String... opts) {
                    List.of(opts).forEach(s -> vmOpts.add(s));
                    return this;
                }


                @Override
                public boolean command() {
                    return command;
                }

                @Override
                public boolean verbose() {
                    return verbose;
                }
                @Override
                public boolean warnings() {
                    return warnings;
                }
                @Override
                public boolean progress() {
                    return progress;
                }

                @Override
                public List<String> vmOpts() {
                    return new ArrayList<>();
                }
            }
        }


        static Config of(Consumer<Builder> javacOptBuilderConsumer) {
            Builder builder = new Builder.Impl();
            javacOptBuilderConsumer.accept(builder);
            return of(builder.command(), builder.warnings(), builder.progress(), builder.verbose(), builder.vmOpts());
        }

        static Config of() {
            return of(false, false, false,false,new ArrayList<>());
        }
    }
}

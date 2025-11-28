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

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

public class JExtract extends Jar {
    final JExtractOptProvider optProvider;
    public void check(){
       // We dont care if the src does not exist unlike a clean jar
    }
    private JExtract(Project.Id id,  Set<Path> exclude, Set<Dependency> dependencies) {
        super(id, exclude, dependencies);
        // We expect the dependencies to include a JextractOptProvider
        var optionalProvider = dependencies.stream().filter(dep -> dep instanceof JExtractOptProvider).map(dep -> (JExtractOptProvider) dep).findFirst();
        this.optProvider = optionalProvider.orElseThrow();
        id.project().add(this);
    }

    @Override
    public Path javaSourcePath() {
        return id.path().resolve("src/main/java");
    }

    @Override
    public boolean build() {
        try {
            id.project().mkdir(javaSourcePath());
            var opts = StringList.of().add(
                    "jextract",
                    "--target-package", id().shortHyphenatedName(),
                    "--output", javaSourcePath().toString()
            );
            optProvider.jExtractOpts(opts);
            optProvider.writeCompilerFlags(id().project().rootPath());
            if (id.project().jextractOpts().command()) {
                System.out.println(opts.toString());
            }
            if (id.project().jextractOpts().progress()) {
                System.out.println("extracting "+id().fullHyphenatedName());
            }
            var result = ForkExec.forkExec(this, false, id.project().rootPath(), opts);
            if (id.project().jextractOpts().verbose()){
                result.stdErrAndOut().forEach(System.out::println);
             }
            super.build();
            return result.status()==0;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public boolean clean(boolean verbose) {
        return false;
    }
    static public JExtract extract(Project.Id id,  Set<Dependency> dependencies) {
        return new JExtract(id,Set.of(), dependencies);
    }

    public  interface Config extends CommonConfig<Config>{
        boolean command();
        boolean warnings();
        boolean progress();
        boolean verbose();



        record ConfigImpl(boolean command ,boolean warnings, boolean progress,  boolean verbose) implements Config {
        }

        static Config of(boolean command, boolean warnings, boolean progress,boolean verbose) {
            return new ConfigImpl(command, warnings,progress,verbose);
        }

        interface Builder extends Config {
            Builder command(boolean f);

            Builder warnings(boolean f);
            Builder progress(boolean f);

            Builder verbose(boolean f);



            class Impl implements Builder {
                boolean command;
                boolean warnings;
                boolean progress;
                boolean verbose;


                @Override
                public Builder command(boolean f) {
                    command = f;
                    return this;
                }
                @Override
                public Builder warnings(boolean f) {
                    warnings = f;
                    return this;
                }
                @Override
                public Builder progress(boolean f) {
                    progress = f;
                    return this;
                }

                @Override
                public Builder verbose(boolean f) {
                    verbose = f;
                    return this;
                }
                @Override
                public boolean command() {
                    return command;
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
                public boolean verbose() {
                    return verbose;
                }

            }
        }


        static Config of(Consumer<Builder> javacOptBuilderConsumer) {
            Builder builder = new Builder.Impl();
            javacOptBuilderConsumer.accept(builder);
            return of(builder.command(), builder.warnings(),builder.progress(),builder.verbose());
        }

        static Config of() {
            return of(false, false, false, false);
        }
    }
}

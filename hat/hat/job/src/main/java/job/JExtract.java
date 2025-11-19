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
import java.util.Set;

public class JExtract extends Jar {
    final JExtractOptProvider optProvider;

    private JExtract(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
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
            var opts = ForkExec.Opts.of("jextract").add(
                    "--target-package", id().shortHyphenatedName(),
                    "--output", javaSourcePath().toString()
            );
            optProvider.jExtractOpts(opts);
            optProvider.writeCompilerFlags(id().project().rootPath());
            id().project().reporter.command(this, opts.toString());
            System.out.println(String.join(" ", opts.toString()));
            id().project().reporter.progress(this, "extracting");
            var result= ForkExec.forkExec(this, id.project().rootPath(),opts);
            result.stdErrAndOut().forEach((line)->{
                id().project().reporter.warning(this, line);
            });
            super.build();
            return result.status()==0;
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public boolean clean() {
        return false;
    }

    static public JExtract extract(Project.Id id, Set<Dependency> dependencies) {
        return new JExtract(id, Set.of(), dependencies);
    }

    static public JExtract extract(Project.Id id, Dependency... dependencies) {
        return new JExtract(id, Set.of(), Set.of(dependencies));
    }
}

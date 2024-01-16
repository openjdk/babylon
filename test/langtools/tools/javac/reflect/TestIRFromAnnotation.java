/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

/*
 * @test
 * @enablePreview
 * @library ../lib
 * @modules jdk.compiler/com.sun.tools.javac.api
 * @summary Smoke test for accessing IR from annotation processors
 * @run main TestIRFromAnnotation
 */

import com.sun.source.util.JavacTask;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.code.op.CoreOps.FuncOp;
import java.lang.reflect.code.op.ExtendedOps;
import java.lang.reflect.code.parser.OpParser;
import java.nio.charset.Charset;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.*;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;

public class TestIRFromAnnotation {

    public static void main(String... args) throws Exception {
        String testSrc = System.getProperty("test.src");
        File baseDir = Path.of(testSrc).toFile();
        new TestIRFromAnnotation().run(baseDir);
    }

    void run(File baseDir) throws Exception {
        for (File file : getAllFiles(List.of(baseDir))) {
            if (!file.exists() || !file.getName().endsWith(".java")) {
                continue;
            }
            analyze(file);
        }
    }

    void analyze(File source) {
        try {
            JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
            JavaFileManager fileManager = compiler.getStandardFileManager(null, null, Charset.defaultCharset());
            JavacTask task = (JavacTask)compiler.getTask(null, fileManager, null,
                    List.of("-proc:only",
                            "--enable-preview",
                            "--source", Integer.toString(SourceVersion.latest().runtimeVersion().feature())),
                    null, List.of(new SourceFile(source)));
            task.setProcessors(List.of(new Processor()));
            task.analyze();
        } catch (Throwable ex) {
            throw new AssertionError("Unexpected exception when analyzing: " + source, ex);
        }
    }

    File[] getAllFiles(List<File> roots) throws IOException {
        long now = System.currentTimeMillis();
        ArrayList<File> buf = new ArrayList<>();
        for (File file : roots) {
            Files.walkFileTree(file.toPath(), new SimpleFileVisitor<Path>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    buf.add(file.toFile());
                    return FileVisitResult.CONTINUE;
                }
            });
        }
        long delta = System.currentTimeMillis() - now;
        System.err.println("All files = " + buf.size() + " " + delta);
        return buf.toArray(new File[buf.size()]);
    }

    static class SourceFile extends SimpleJavaFileObject {

        private final File file;
        protected SourceFile(File file) {
            super(file.toURI(), Kind.SOURCE);
            this.file = file;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException {
            return Files.readString(file.toPath());
        }
    }

    public static class Processor extends JavacTestingAbstractProcessor {

        public boolean process(Set<? extends TypeElement> annotations,
                               RoundEnvironment roundEnvironment) {
            class Scan extends ElementScanner<Void,Void> {
                @Override
                public Void visitExecutable(ExecutableElement e, Void p) {
                    IR ir = e.getAnnotation(IR.class);
                    if (ir == null) {
                        return null; // skip
                    }
                    Optional<Object> body = elements.getBody(e);
                    if (!body.isPresent()) {
                        throw new AssertionError(String.format("No body found in method %s annotated with @IR",
                                toMethodString(e)));
                    }
                    String actualOp = ((FuncOp)body.get()).toText();
                    String expectedOp = OpParser.fromString(ExtendedOps.FACTORY, ir.value()).get(0).toText();
                    if (!actualOp.equals(expectedOp)) {
                        throw new AssertionError(String.format("Bad IR found in %s:\n%s\nExpected:\n%s",
                                toMethodString(e), actualOp, expectedOp));
                    }
                    return null;
                }
            }
            Scan scan = new Scan();
            for (Element e : roundEnvironment.getRootElements()) {
                scan.scan(e);
            }
            return true;
        }
    }

    static String toMethodString(ExecutableElement e) {
        return e.getEnclosingElement() + "." + e;
    }

}

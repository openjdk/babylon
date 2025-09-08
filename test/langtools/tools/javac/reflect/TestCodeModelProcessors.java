/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
 *  @test
 *  @summary check that annotation processor that depends on jdk.incubator.code can run normally
 *  @library /tools/lib
 *  @modules jdk.compiler/com.sun.tools.javac.api
 *           jdk.compiler/com.sun.tools.javac.main
 *           jdk.jlink
 *  @build toolbox.ToolBox toolbox.JavacTask toolbox.JarTask
 *  @run main/othervm TestCodeModelProcessors
 */

import java.io.IOException;
import java.util.List;
import java.nio.file.Files;
import java.nio.file.Path;

import toolbox.JarTask;
import toolbox.JavacTask;
import toolbox.Task;
import toolbox.Task.OutputKind;
import toolbox.TestRunner;
import toolbox.ToolBox;

public class TestCodeModelProcessors extends TestRunner {
    public static void main(String... args) throws Exception {
        new TestCodeModelProcessors().run();
    }

    TestCodeModelProcessors() {
        super(System.out);
    }

    ToolBox tb = new ToolBox();

    Path pluginJar;
    Path classes;
    Path mclasses;

    void run() throws Exception {
        Path src = Path.of("src");
        tb.writeJavaFiles(src,
                """
                        package p;

                        import javax.annotation.processing.*;
                        import jdk.incubator.code.*;
                        import java.util.*;

                        public class TestProcessor extends AbstractProcessor {

                            @Override
                            public void init(ProcessingEnvironment processingEnv) {
                                try {
                                    var processorClass = TestProcessor.class;
                                    var testMethod = processorClass.getDeclaredMethod("test");
                                    var op = Op.ofMethod(testMethod);
                                    if (!op.isPresent()) {
                                        throw new AssertionError("BAD");
                                    } else {
                                        test();
                                    }
                                } catch (Throwable ex) {
                                    throw new AssertionError(ex);
                                }
                            }

                            @Override
                            public boolean process(Set<? extends javax.lang.model.element.TypeElement> annotations, RoundEnvironment roundEnv) {
                                return true;
                            }

                            @CodeReflection
                            void test() {
                                System.out.println("SUCCESS");
                            }
                        }
                        """);

        Path msrc = Path.of("msrc");
        tb.writeJavaFiles(msrc,
                """
                          module m {
                              requires jdk.compiler;
                              requires jdk.incubator.code;
                              provides javax.annotation.processing.Processor with p.TestProcessor;
                          }
                          """);

        classes = Files.createDirectories(Path.of("classes"));
        new JavacTask(tb)
                .options("--add-modules", "jdk.incubator.code")
                .outdir(classes)
                .files(tb.findJavaFiles(src))
                .run()
                .writeAll();

        tb.writeFile(classes.resolve("META-INF").resolve("services").resolve("javax.annotation.processing.Processor"),
                "p.TestProcessor\n");

        pluginJar = Path.of("plugin.jar");
        new JarTask(tb, pluginJar)
                .baseDir(classes)
                .files(".")
                .run();

        mclasses = Files.createDirectories(Path.of("mclasses"));
        new JavacTask(tb)
                .options("--add-modules", "jdk.incubator.code")
                .outdir(mclasses)
                .sourcepath(msrc, src)
                .files(tb.findJavaFiles(msrc))
                .run()
                .writeAll();

        Path hw = Path.of("hw");
        tb.writeJavaFiles(hw,
                """
                        import jdk.incubator.code.*;

                        public class HelloWorld {
                            @CodeReflection
                            void testAnnos() { }
                        }
                        """);

        runTests(m -> new Object[] { Path.of(m.getName()) });
    }

    @Test
    public void testClassPath(Path base) throws Exception {
        List<String> stdout = new JavacTask(tb)
                .classpath(classes)
                .options("--processor-path", classes.toString())
                .options("-processor", "p.TestProcessor")
                .options("--add-modules", "jdk.incubator.code")
                .outdir(Files.createDirectories(base.resolve("out")))
                .files(tb.findJavaFiles(Path.of("hw")))
                .run()
                .writeAll()
                .getOutputLines(OutputKind.STDOUT);
        tb.checkEqual(stdout, List.of("SUCCESS"));
    }

    @Test
    public void testClassPathJar(Path base) throws Exception {
        List<String> stdout = new JavacTask(tb)
                .classpath(pluginJar)
                .options("--add-modules", "jdk.incubator.code")
                .outdir(Files.createDirectories(base.resolve("out")))
                .files(tb.findJavaFiles(Path.of("hw")))
                .run()
                .writeAll()
                .getOutputLines(Task.OutputKind.STDOUT);
        tb.checkEqual(stdout, List.of("SUCCESS"));
    }

    @Test
    public void testModulePath(Path base) throws IOException {
        List<String> stdout = new JavacTask(tb)
                .options("--processor-module-path", mclasses.toString())
                .options("--add-modules", "jdk.incubator.code")
                .outdir(Files.createDirectories(base.resolve("out")))
                .files(tb.findJavaFiles(Path.of("hw")))
                .run()
                .writeAll()
                .getOutputLines(Task.OutputKind.STDOUT);
        tb.checkEqual(stdout, List.of("SUCCESS"));
    }
}


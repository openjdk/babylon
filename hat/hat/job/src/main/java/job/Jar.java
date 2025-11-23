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

import com.sun.source.util.JavacTask;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;
import java.util.stream.Collectors;

public class Jar extends DependencyImpl<Jar> implements Dependency.Buildable, Dependency.WithPath, Dependency.ExecutableJar {
    final Set<Path> exclude;
    final JavacOpts javacOpts;
    public interface StringListBuilder {
        List<String> strings();
        StringListBuilder add(String ...s);
        StringListBuilder add(List<String> list);
        StringListBuilder addIf(boolean c, String ...s);
        StringListBuilder addIf(boolean c, List<String> list);
        static StringListBuilder of(){
            record StringListBuilderImpl (List<String> strings) implements StringListBuilder{
                @Override
                public StringListBuilder add(String ...s) {
                    return add(List.of(s));
                }
                @Override
                public StringListBuilder add(List<String> list) {
                    strings.addAll(list);
                    return this;
                }
                @Override
                public StringListBuilder addIf(boolean c, List<String> list) {
                    if (c){
                        strings.addAll(list);
                    }
                    return this;
                }
                @Override
                public StringListBuilder addIf(boolean c, String ...s) {
                   return addIf(c, List.of(s));
                }
            }
            return new StringListBuilderImpl(new ArrayList<>());
        }
    }
    public interface JavacOpts{
        List<String> vmOpts();
        List<String> opts();

        static JavacOpts of(List<String> vmOpts, List<String> opts){
            record JavacOptsImpl(List<String> vmOpts, List<String> opts) implements JavacOpts{
            }
            return new JavacOptsImpl(vmOpts,opts);
        }
        static JavacOpts of(Consumer<StringListBuilder> v){
            StringListBuilder vmOptBuilder = StringListBuilder.of();
            StringListBuilder optBuilder = StringListBuilder.of();
            v.accept(vmOptBuilder);
            return of(vmOptBuilder.strings(),optBuilder.strings());
        }

        static JavacOpts of(Consumer<StringListBuilder> v, Consumer<StringListBuilder> o){
            StringListBuilder vmOptBuilder = StringListBuilder.of();
            StringListBuilder optBuilder = StringListBuilder.of();
            v.accept(vmOptBuilder);
            o.accept(optBuilder);
            return of(vmOptBuilder.strings(),optBuilder.strings());
        }

        static JavacOpts of(){
            return of(new ArrayList<>(),new ArrayList<>());
        }
    }

    public interface JavaOpts{
        List<String> opts();
        List<String> args();

        static JavaOpts of(List<String> opts, List<String> args){
            record JavaOptsImpl(List<String>opts, List<String> args) implements JavaOpts{
            }
            return new JavaOptsImpl(opts,args);
        }
        static JavaOpts of(Consumer<StringListBuilder> v){
            StringListBuilder vmOptBuilder = StringListBuilder.of();
            StringListBuilder optBuilder = StringListBuilder.of();
            v.accept(vmOptBuilder);
            return of(vmOptBuilder.strings(),optBuilder.strings());
        }

        static JavaOpts of(Consumer<StringListBuilder> v, Consumer<StringListBuilder> o){
            StringListBuilder vmOptBuilder = StringListBuilder.of();
            StringListBuilder optBuilder = StringListBuilder.of();
            v.accept(vmOptBuilder);
            o.accept(optBuilder);
            return of(vmOptBuilder.strings(),optBuilder.strings());
        }

        static JavaOpts of(){
            return of(new ArrayList<>(),new ArrayList<>());
        }
    }

    protected Jar(Project.Id id, JavacOpts javacOpts,Set<Path> exclude, Set<Dependency> dependencies) {
        super(id, dependencies);
        this.javacOpts = javacOpts;
        this.exclude = exclude;
        if (id.path() != null && !Files.exists(id.path())) {
            System.err.println("The path does not exist: " + id.path());
        }
        if (!Files.exists(javaSourcePath())) {
            var jsp = javaSourcePath();
            System.out.println("Failed to find java source " + jsp + " path for " + id.shortHyphenatedName());
        }
        id.project().add(this);
    }
    public static Jar of(Project.Id id, JavacOpts javacOpts, Set<Path> exclude, Set<Dependency> dependencies) {
        return new Jar(id, javacOpts,exclude, dependencies);
    }

    public static Jar of(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
        return new Jar(id, JavacOpts.of(), exclude, dependencies);
    }
    public static Jar of(Project.Id id, JavacOpts javacOpts, Set<Dependency> dependencies) {
        return new Jar(id, javacOpts,Set.of(), dependencies);
    }


    public static Jar of(Project.Id id, Set<Dependency> dependencies) {
        return new Jar(id,JavacOpts.of(), Set.of(), dependencies);
    }

    public static Jar of(Project.Id id, Set<Path> exclude, Dependency... dependencies) {
        return of(id, exclude, Set.of(dependencies));
    }

    public static Jar of(Project.Id id, Dependency... dependencies) {
        return of(id, Set.of(), Set.of(dependencies));
    }

    public static class JavaSource extends SimpleJavaFileObject {
        Path path;

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            try {
                return Files.readString(Path.of(toUri()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        JavaSource(Path path) {
            super(path.toUri(), Kind.SOURCE);
            this.path = path;
        }
    }

    public Path jarFile() {
        return id().project().buildPath().resolve(id().fullHyphenatedName() + ".jar");
    }

    @Override
    public List<Path> generatedPaths() {
        throw new IllegalStateException("who called me");
    }


    @Override
    public boolean clean() {
        id().project().clean(null, classesDir(), jarFile());
        return true;
    }

    @Override
    public boolean build() {
        List<String> opts = new ArrayList<>();
        opts.addAll(id().project().javacOpts().vmOpts());
        opts.addAll(javacOpts.vmOpts());
        opts.addAll(List.of(
                "-d", classesDirName()
                )
        );
        Dag dag = new Dag(dependencies());
        var deps = classPath(dag.ordered());

        if (!deps.isEmpty()) {
            opts.add("--class-path=" + deps);
        }
        opts.addAll(id.project().javacOpts().opts());
        opts.addAll(javacOpts.opts());
        opts.add("--source-path=" + javaSourcePathName());
        //System.out.println("javac opts "+ String.join(" ", opts));
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();

        id().project().clean(this, classesDir());

        if (Files.exists(javaSourcePath())) {
            try (var files = Files.walk(javaSourcePath())) {
                var listOfSources = files.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".java") && !exclude.contains(p)).map(JavaSource::new).toList();
                id().project().reporter.command(this, "javac " +
                        String.join(" ", opts) + " " + String.join(" ",
                        listOfSources.stream().map(JavaSource::getName).collect(Collectors.toList())));


                var diagnosticListener = new DiagnosticListener<JavaFileObject>() {
                    @Override
                    public void report(Diagnostic<? extends JavaFileObject> diagnostic) {
                        if (diagnostic.getKind() == Diagnostic.Kind.ERROR) {
                            id().project().reporter.error(Jar.this, diagnostic.toString());
                        } else if (diagnostic.getKind() == Diagnostic.Kind.WARNING) {
                            id().project().reporter.warning(Jar.this, diagnostic.toString());
                        } else if (diagnostic.getKind() == Diagnostic.Kind.MANDATORY_WARNING) {
                            id().project().reporter.warning(Jar.this, "!!" + diagnostic.toString());
                        } else if (diagnostic.getKind() == Diagnostic.Kind.NOTE) {
                            id().project().reporter.note(Jar.this, diagnostic.toString());
                        } else {
                            id().project().reporter.warning(Jar.this, diagnostic.getKind() + ":" + diagnostic.toString());
                        }
                    }
                };
                ((JavacTask) javac.getTask(
                        new PrintWriter(System.err),
                        javac.getStandardFileManager(diagnosticListener, null, null),
                        diagnosticListener,
                        opts,
                        null,
                        listOfSources
                )).generate().forEach(gc ->
                        id.project().reporter.note(this, gc.getName())
                );

                List<Path> dirsToJar = new ArrayList<>(List.of(classesDir()));
                if (Files.exists(javaResourcePath())) {
                    dirsToJar.add(javaResourcePath());
                }

                Manifest manifest = new Manifest();
                Attributes mainAttributes = manifest.getMainAttributes();
                mainAttributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
               // mainAttributes.put(Attributes.Name.MAIN_CLASS,   id().shortHyphenatedName()+".Main");
               // mainAttributes.put(Attributes.Name.IMPLEMENTATION_VENDOR, "HAT's Java Opinionated Builder (JOB)");
                var jarStream = new JarOutputStream(Files.newOutputStream(jarFile()), manifest);
                record RootAndPath(Path root, Path path) {
                }
                id().project().reporter.command(this, "jar cvf " + jarFile() + " " +
                        String.join(dirsToJar.stream().map(Path::toString).collect(Collectors.joining(" "))));
                id().project().reporter.progress(this, "compiled " + listOfSources.size() + " file" + (listOfSources.size() > 1 ? "s" : "") + " to " + jarFile().getFileName());

                dirsToJar.forEach(r -> {
                    try {

                        Files.walk(r)
                                .filter(p -> !Files.isDirectory(p))
                                .map(p -> new RootAndPath(r, p))
                                .sorted(Comparator.comparing(RootAndPath::path))
                                .forEach(
                                        rootAndPath -> {
                                            try {
                                                var entry = new JarEntry(rootAndPath.root.relativize(rootAndPath.path).toString());
                                                entry.setTime(Files.getLastModifiedTime(rootAndPath.path()).toMillis());
                                                jarStream.putNextEntry(entry);
                                                Files.newInputStream(rootAndPath.path()).transferTo(jarStream);
                                                jarStream.closeEntry();
                                            } catch (IOException e) {
                                                throw new RuntimeException(e);
                                            }
                                        });


                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                });

                jarStream.finish();
                jarStream.close();
                return true;
            } catch (Exception e) {
                //   println(e.getMessage());
                throw new RuntimeException(e);
            }
        } else {
            return true;
        }
    }

    protected String classPath(Set<Dependency> dependencies) {
        return String.join(":", dependencies.stream().filter(p ->
                p instanceof Jar).map(a -> (Jar) a).map(Jar::jarFileName).toList());
    }

    protected String classPathWithThisLast(Set<Dependency> dependencies) {
        Set<Dependency> all = new LinkedHashSet<>(dependencies);
        all.remove(this);
        all.add(this);
        return String.join(":", all.stream().filter(p ->
                p instanceof Jar).map(a -> (Jar) a).map(Jar::jarFileName).toList());
    }

    private Path classesDir() {
        return id().project().buildPath().resolve(id().fullHyphenatedName() + ".classes");
    }

    private String classesDirName() {
        return classesDir().toString();
    }

    private String jarFileName() {
        return jarFile().toString();
    }

    private Path javaResourcePath() {
        return id().path().resolve("src/main/resources");

    }

    private String javaResourcePathName() {
        return javaResourcePath().toString();
    }

    private String javaSourcePathName() {
        return javaSourcePath().toString();
    }

    protected Path javaSourcePath() {
        return id().path().resolve("src/main/java");
    }

    @Override
    public boolean run(String mainClassName, Set<Dependency> depsInOrder, List<String> vmOpts, List<String> args) {
        ForkExec.Opts opts = ForkExec.Opts.of(ProcessHandle.current()
                .info()
                .command()
                .orElseThrow()).add(vmOpts);
        opts.add(
                "--class-path", classPathWithThisLast(depsInOrder),
                "-Djava.library.path=" + id().project().buildPath()
        );
        vmOpts.forEach(opts::add);
        opts.add(mainClassName);
        args.forEach(opts::add);
        id().project().reporter.command(this, opts.toString());
        System.out.println(String.join(" ", opts.toString()));
        id().project().reporter.progress(this, "running");
        var result = ForkExec.forkExec(this, id().project().rootPath(), opts);
        result.stdErrAndOut().forEach((line) -> {
            id().project().reporter.warning(this, line);
        });
        if (result.status() != 0) {
            System.out.println("Java failed to execute, is a valid java in your path ? " + id().fullHyphenatedName());
        }
        return result.status() == 0;
    }
    public boolean run(String mainClassName, Set<Dependency> depsInOrder, JavaOpts javaOpts) {
        return run(mainClassName,depsInOrder,javaOpts.opts(),javaOpts.args());

    }
}

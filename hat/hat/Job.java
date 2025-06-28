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


import com.sun.source.util.JavacTask;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class Job {
    @FunctionalInterface
    public interface Progress extends BiConsumer<Dependency, String> {
        void accept(Dependency a, String s);
    }

    public interface Dependency {
        Project.Id id();

        Set<Dependency> dependencies();

        interface WithPath extends Dependency {
        }

        interface Buildable extends Dependency {
            boolean build();

            boolean clean();

            List<Path> generatedPaths();
        }

        interface Executable extends Dependency {
        }

        interface ExecutableJar extends Executable {
            boolean run(String mainClassName, Set<Dependency> depsInOrder, List<String> args);
        }

        interface Runnable extends Executable {
            boolean run();
        }

        interface Optional extends Dependency {
            boolean isAvailable();
        }
    }


    public static abstract class AbstractArtifact<T extends AbstractArtifact<T>> implements Dependency {
        protected final Project.Id id;

        @Override
        public Project.Id id() {
            return id;
        }

        final private Set<Dependency> dependencies = new LinkedHashSet<>();

        @Override
        public Set<Dependency> dependencies() {
            return dependencies;
        }

        AbstractArtifact(Project.Id id, Set<Dependency> dependencies) {
            this.id = id;
            this.dependencies.addAll(dependencies);
        }
    }

    public static abstract class AbstractArtifactWithPath<T extends AbstractArtifact<T>> extends AbstractArtifact<T>
            implements Dependency.WithPath {

        AbstractArtifactWithPath(Project.Id id, Set<Dependency> dependencies) {
            super(id, dependencies);
            if (!Files.exists(id.path())) {
                System.err.println("The path does not exist: " + id.path());

            }
        }
    }

    public static class Project {

        public record Id(Project project, String projectName, String hyphenatedName, String version, Path path,
                         String name) {
        }

        private static Id id(Project project, String hyphenatedName) {
            int lastIndex = hyphenatedName.lastIndexOf('-');
            var version = hyphenatedName.substring(lastIndex + 1);
            String[] splitString = hyphenatedName.substring(0, lastIndex).split("-");
            var runName = "";
            var dirName = "";
            if (splitString.length == 3) {
                runName = splitString[1] + "-" + splitString[2];
                dirName = splitString[0] + "s/" + splitString[1] + "/" + splitString[2];
            } else if (splitString.length == 2) {
                runName = splitString[1];
                dirName = splitString[0] + "s/" + splitString[1];
            } else if (splitString.length == 1) {
                runName = splitString[0];
                dirName = splitString[0];
            }
            var id = new Id(project, project.name(), hyphenatedName, version, project.rootPath().resolve(dirName), runName);
            return id;
        }

        Id id(String id) {
            return id(this, id);
        }


        public static class Dag {
            static void recurse(Map<Dependency, Set<Dependency>> map, Dependency from) {
                var set = map.computeIfAbsent(from, _ -> new LinkedHashSet<>());
                var deps = from.dependencies();
                deps.forEach(dep -> {
                    set.add(dep);
                    recurse(map, dep);
                });
            }

            static Set<Dependency> processOrder(Set<Dependency> jars) {
                Map<Dependency, Set<Dependency>> map = new LinkedHashMap<>();
                Set<Dependency> ordered = new LinkedHashSet<>();
                jars.forEach(jar -> recurse(map, jar));
                while (!map.isEmpty()) {
                    var leaves = map.entrySet().stream()
                            .filter(e -> e.getValue().isEmpty())    // if this entry has zero dependencies
                            .map(Map.Entry::getKey)                 // get the key
                            .collect(Collectors.toSet());
                    map.forEach((k, v) ->
                            leaves.forEach(v::remove)
                    );
                    leaves.forEach(leaf -> {
                        map.remove(leaf);
                        ordered.add(leaf);
                    });
                }
                return ordered;
            }

            static Set<Dependency> build(Set<Dependency> jars) {
                var ordered = processOrder(jars);
                ordered.stream().filter(d -> d instanceof Dependency.Buildable).map(d -> (Dependency.Buildable) d).forEach(Dependency.Buildable::build);
                return ordered;
            }

            static Set<Dependency> clean(Set<Dependency> jars) {
                var ordered = processOrder(jars);
                ordered.stream().filter(d -> d instanceof Dependency.Buildable).map(d -> (Dependency.Buildable) d).forEach(Dependency.Buildable::build);
                return ordered;
            }

        }

        private final Path rootPath;
        private final Path buildPath;
        private final Path confPath;
        private final Progress progress;
        private final Map<String, Dependency> artifacts = new LinkedHashMap<>();

        public String name() {
            return rootPath().getFileName().toString();
        }

        public Path rootPath() {
            return rootPath;
        }

        public Path buildPath() {
            return buildPath;
        }

        public Path confPath() {
            return confPath;
        }

        public Project(Path root, Progress progress) {
            this.rootPath = root;
            if (!Files.exists(root)) {
                throw new IllegalArgumentException("Root path for project does not exist: " + root);
            }
            this.buildPath = root.resolve("build");
            this.confPath = root.resolve("conf");
            this.progress = progress;
        }

        public Project(Path root) {
            this(root, (a, s) -> System.out.println(a.id().project().name() + ":" + a.id().name() + ":" + s));
        }

        public Dependency add(Dependency dependency) {
            artifacts.put(dependency.id().hyphenatedName, dependency);
            return dependency;
        }

        public Dependency getArtifact(String dependency) {
            return artifacts.get(dependency);
        }

        public void rmdir(Path... paths) {
            for (Path path : paths) {
                //  System.out.println("rm -rf "+path.getFileName().toString());
                if (Files.exists(path)) {
                    try (var files = Files.walk(path)) {
                        files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
                    } catch (Throwable t) {
                        throw new RuntimeException(t);
                    }
                }
            }
        }

        public void clean(Path... paths) {
            for (Path path : paths) {
                if (Files.exists(path)) {
                    // System.out.println("rm -rf "+path.getFileName().toString());
                    // System.out.println("mkdir -p "+path.getFileName().toString());
                    try (var files = Files.walk(path)) {
                        files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
                        mkdir(path);
                    } catch (Throwable t) {
                        throw new RuntimeException(t);
                    }
                }
            }
        }

        public void mkdir(Path... paths) {
            for (Path path : paths) {
                if (!Files.exists(path)) {
                    try {
                        Files.createDirectories(path);
                    } catch (Throwable t) {
                        throw new RuntimeException(t);
                    }
                }
            }
        }

        public Set<Dependency> clean(Set<Dependency> dependencies) {
            return Dag.clean(dependencies);
        }

        public void clean(List<String> names) {
            if (names.isEmpty()) {
                rmdir(buildPath());
            } else {
                clean(names.stream().map(this::getArtifact).collect(Collectors.toSet()));
            }
        }

        public Set<Dependency> build(Set<Dependency> dependencies) {
            return Dag.build(dependencies);
        }

        public Set<Dependency> build(Dependency... dependencies) {
            return build(Set.of(dependencies));
        }

        public Set<Dependency> build(List<String> names) {
            if (names.isEmpty()) {
                return build(new HashSet<>(artifacts.values()));
            } else {
                return build(names.stream().map(this::getArtifact).collect(Collectors.toSet()));
            }
        }

        void start(String... argArr) throws IOException, InterruptedException {
            var args = new ArrayList<>(List.of(argArr));

            Map<String, String> opts = Map.of(
                    "bld", "Will Bld",
                    "help", """
                             help: This list
                              bld: ...buildables | all if none
                                   bld
                                   bld ffi-opencl
                              run: [ffi|my|seq]-[opencl|java|cuda|mock|hip] runnable (i.e has name.Main class)
                                   run ffi-opencl mandel
                                   run ffi-openc nbody
                            clean: ...buildables | all if none
                                   clean
                                   clean ffi-opencl
                            """,
                    "clean", "Will clean",
                    "run", "Will run"
            );
            record Action(String name, String help, List<String> args) {
                int size() {
                    return args.size();
                }

                boolean isEmpty() {
                    return args.isEmpty();
                }

                String get() {
                    var got = (size() > 0) ? args.removeFirst() : null;
                    return got;
                }

                String str() {
                    return name + " '" + String.join(" ", args) + "'";
                }

            }

            List<Action> actions = new ArrayList<>();
            while (!args.isEmpty()) {
                String arg = args.removeFirst();
                if (opts.containsKey(arg)) {
                    List<String> subList = new ArrayList<>();
                    while (!args.isEmpty() &&
                            args.getFirst() instanceof String next
                            && !opts.containsKey(next)) {
                        subList.add(args.removeFirst());
                    }
                    actions.add(new Action(arg, opts.get(arg), subList));
                } else {
                    System.err.println("What " + arg + " " + String.join(" ", args));
                }
            }
            if (actions.stream().anyMatch(a -> a.name.equals("help"))) {
                actions.forEach(action ->
                        System.out.println(action.help)
                );
            } else {
                for (var action : actions) {
                    switch (action.name()) {
                        case "clean" -> clean(action.args);
                        case "bld" -> build(action.args);
                        case "run" -> {
                            if (action.get() instanceof String backendName && !action.isEmpty() && getArtifact("backend-" + backendName + "-1.0") instanceof Jar backend) {
                                if (action.get() instanceof String runnableName && getArtifact("example-" + runnableName + "-1.0") instanceof Dependency.ExecutableJar runnable) {
                                    runnable.run(runnable.id().name() + ".Main", build(runnable, backend), args);
                                } else {
                                    System.out.println("Failed to find runnable ");
                                }
                            } else {
                                System.out.println("Failed to find backend !");
                            }
                        }
                        default -> {
                        }
                    }
                }
            }

        }
    }


    public static class Jar extends AbstractArtifactWithPath<Jar> implements Dependency.Buildable {
        public interface JavacProgress extends Progress {

            default void javacCommandLine(Dependency a, List<String> opts, List<JavaSource> sources) {
                accept(a, "javac " + String.join(" ", opts) + " " + String.join(" ", sources.stream().map(JavaSource::getName).collect(Collectors.toList())));
            }

            default void javacInfo(Dependency a, String s) {
                accept(a, "JAVAC : I" + s);
            }

            default void javacProgress(Dependency a, String s) {
                accept(a, "JAVAC : " + s);
            }

            default void javacError(Dependency a, String s) {
                accept(a, "JAVAC : !!!" + s);
                throw new RuntimeException(s);
            }

            default void javacWarning(Dependency a, String s) {
                accept(a, "JAVAC : W " + s);
            }

            default void javacClean(Dependency a, Path... paths) {
                accept(a, "clean " + String.join(" ", Arrays.stream(paths).map(Path::toString).toList()));
            }

            default void javacNote(Dependency a, String s) {
                accept(a, "JAVAC :" + s);
            }

            default void javacVerbose(Dependency a, String s) {
                accept(a, "JAVAC :" + s);
            }

            default void javacCreatedClass(Dependency a, String s) {
                accept(a, "JAVAC_CREATED_CLASS :" + s);
            }

            static JavacProgress adapt(Project.Id id) {
                return (id.project().progress instanceof JavacProgress progress) ? progress : new JavacProgress() {
                    @Override
                    public void accept(Dependency a, String s) {
                        id.project().progress.accept(a, s);
                    }
                };
            }

        }

        public interface JarProgress extends Progress {
            default void jarProgress(Dependency a, String s) {
                accept(a, "JAR :" + s);
            }

            default void jarInfo(Dependency a, String s) {
                accept(a, "JAR : I" + s);
            }

            default void jarCommandLine(Dependency a, Path path, List<Path> paths) {
                accept(a, "jar cvf " + path + " " + String.join(paths.stream().map(Path::toString).collect(Collectors.joining(" "))));
            }

            default void jarClean(Dependency a, Path... paths) {
                accept(a, "clean " + String.join(" ", Arrays.stream(paths).map(Path::toString).toList()));
            }

            default void jarError(Dependency a, String s) {
                accept(a, "JAR : !!!" + s);
                throw new RuntimeException(s);
            }

            default void jarWarning(Dependency a, String s) {
                accept(a, "JAR : W " + s);
            }

            default void jarNote(Dependency a, String s) {
                accept(a, "JAR :" + s);
            }

            static JarProgress adapt(Project.Id id) {
                return (id.project.progress instanceof JarProgress progress) ? progress : new JarProgress() {
                    @Override
                    public void accept(Dependency a, String s) {
                        id.project.progress.accept(a, s);
                    }
                };
            }
        }

        final Set<Path> exclude;

        private Jar(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
            super(id, dependencies);
            this.exclude = exclude;

            if (!Files.exists(javaSourcePath())) {
                var jsp = javaSourcePath();
                System.out.println("Failed to find java source " + jsp + " path for " + id.name());
            }
            id.project.add(this);
        }

        public static Jar of(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
            return new Jar(id, exclude, dependencies);
        }

        public static Jar of(Project.Id id, Set<Dependency> dependencies) {
            return new Jar(id, Set.of(), dependencies);
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
            return id().project().buildPath().resolve(id().hyphenatedName() + ".jar");
        }

        @Override
        public List<Path> generatedPaths() {
            throw new IllegalStateException("who called me");
        }


        @Override
        public boolean clean() {
            RunnableJar.JavacProgress.adapt(id()).javacClean(this, classesDir(), jarFile());
            RunnableJar.JarProgress.adapt(id()).jarClean(this, classesDir(), jarFile());
            id().project().clean(classesDir(), jarFile());
            return true;
        }

        @Override
        public boolean build() {
            List<String> opts = new ArrayList<>(
                    List.of(
                            "--source=26",
                            "--enable-preview",
                            "--add-modules=jdk.incubator.code",
                            "--add-exports=jdk.incubator.code/jdk.incubator.code.dialect.java.impl=ALL-UNNAMED",
                            "-g",
                            "-d", classesDirName()
                    ));
            var deps = classPath(Project.Dag.processOrder(dependencies()));
            if (!deps.isEmpty()) {
                opts.addAll(List.of(
                        "--class-path=" + deps
                ));
            }
            opts.addAll(List.of(
                            "--source-path=" + javaSourcePathName()
                    )
            );

            JavacProgress javacProgress = JavacProgress.adapt(id());

            JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
            javacProgress.javacClean(this, classesDir());
            id().project().clean(classesDir());

            if (Files.exists(javaSourcePath())) {
                try (var files = Files.walk(javaSourcePath())) {
                    var listOfSources = files.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".java") && !exclude.contains(p)).map(JavaSource::new).toList();

                    javacProgress.javacCommandLine(this, opts, listOfSources);
                    var diagnosticListener = new DiagnosticListener<JavaFileObject>() {
                        @Override
                        public void report(Diagnostic<? extends JavaFileObject> diagnostic) {
                            if (diagnostic.getKind() == Diagnostic.Kind.ERROR) {
                                javacProgress.javacError(Jar.this, diagnostic.toString());
                            } else if (diagnostic.getKind() == Diagnostic.Kind.WARNING) {
                                javacProgress.javacWarning(Jar.this, diagnostic.toString());
                            } else if (diagnostic.getKind() == Diagnostic.Kind.MANDATORY_WARNING) {
                                javacProgress.javacWarning(Jar.this, "!!" + diagnostic.toString());
                            } else if (diagnostic.getKind() == Diagnostic.Kind.NOTE) {
                                javacProgress.javacNote(Jar.this, diagnostic.toString());
                            }
                            javacProgress.javacProgress(Jar.this, diagnostic.getKind() + ":" + diagnostic.toString());
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
                            javacProgress.javacCreatedClass(this, gc.getName())
                    );

                    List<Path> dirsToJar = new ArrayList<>(List.of(classesDir()));
                    if (Files.exists(javaResourcePath())) {
                        dirsToJar.add(javaResourcePath());
                    }
                    var jarStream = new JarOutputStream(Files.newOutputStream(jarFile()));
                    JarProgress jarProgress = JarProgress.adapt(id());

                    record RootAndPath(Path root, Path path) {
                    }
                    dirsToJar.forEach(r -> {
                        try {
                            jarProgress.jarCommandLine(this, jarFile(), dirsToJar);

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
            return id().project().buildPath().resolve(id().hyphenatedName() + ".classes");
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
    }

    public static class RunnableJar extends Jar implements Dependency.ExecutableJar {
        public interface JavaProgress extends Progress {
            default void javaProgress(Dependency a, String s) {
                accept(a, "JAVA :" + s);
            }

            default void javaInfo(Dependency a, String s) {
                accept(a, "JAVA : I" + s);
            }

            default void javaCommandLine(Dependency a, String s) {
                accept(a, "JAVA : " + s);
            }

            default void javaError(Dependency a, String s) {
                accept(a, "JAVA : !!!" + s);
                throw new RuntimeException(s);
            }

            default void javaWarning(Dependency a, String s) {
                accept(a, "JAVA : W " + s);
            }

            default void javaNote(Dependency a, String s) {
                accept(a, "JAVA :" + s);
            }

            static JavaProgress adapt(Project.Id id) {
                return (id.project().progress instanceof JavaProgress progress) ? progress : new JavaProgress() {
                    @Override
                    public void accept(Dependency a, String s) {
                        id.project().progress.accept(a, s);
                    }
                };
            }
        }

        private RunnableJar(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
            super(id, exclude, dependencies);
            id.project.add(this);
        }

        static public RunnableJar of(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
            return new RunnableJar(id, exclude, dependencies);
        }

        static public RunnableJar of(Project.Id id, Set<Path> exclude, Dependency... dependencies) {
            return of(id, exclude, Set.of(dependencies));
        }

        static public RunnableJar of(Project.Id id, Set<Dependency> dependencies) {
            return new RunnableJar(id, Set.of(), dependencies);
        }

        static public RunnableJar of(Project.Id id, Dependency... dependencies) {
            return of(id, Set.of(), Set.of(dependencies));
        }

        @Override
        public List<Path> generatedPaths() {
            throw new IllegalStateException("who called me");
        }

        @Override
        public boolean run(String mainClassName, Set<Dependency> depsInOrder, List<String> args) {
            JavaProgress javaProgress = JavaProgress.adapt(id());
            List<String> opts = new ArrayList<>();
            opts.addAll(List.of(
                    "/Users/grfrost/github/babylon-grfrost-fork/build/macosx-aarch64-server-release/jdk/bin/java",
                    "--enable-preview",
                    "--enable-native-access=ALL-UNNAMED"));
            if (id().name().equals("nbody")) {
                opts.addAll(List.of(
                        "-XstartOnFirstThread"
                ));
            }
            opts.addAll(List.of(
                    "--add-exports=jdk.incubator.code/jdk.incubator.code.dialect.java.impl=ALL-UNNAMED", // for OpRenderer
                    "--class-path", classPathWithThisLast(depsInOrder),
                    "-Djava.library.path=" + id().project().buildPath,
                    mainClassName
            ));
            opts.addAll(args);
            javaProgress.javaCommandLine(this, String.join(" ", opts));
            try {
                var process = new ProcessBuilder().command(opts).redirectOutput(ProcessBuilder.Redirect.INHERIT).start();
                process.waitFor();
                return process.exitValue() == 0;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }


    public static class CMake extends AbstractArtifactWithPath<CMake> implements Dependency.Buildable {


        public interface CMakeProgress extends Progress {

            default void cmakeProgress(Dependency a, String s) {
                accept(a, "CMAKE :" + s);
            }

            default void cmakeInfo(Dependency a, String s) {
                accept(a, "CMAKE :" + s);
            }

            default void cmakeError(Dependency a, String s) {
                accept(a, "CMAKE :!!" + s);
            }

            default void cmakeCommandLine(Dependency a, String s) {
                accept(a, "CMAKE :" + s);
            }

            default void cmakeVerbose(Dependency a, String s) {
                accept(a, "CMAKE :     " + s);
            }

            static CMakeProgress adapt(Project.Id id) {
                return (id.project().progress instanceof CMakeProgress progress) ? progress : new CMakeProgress() {
                    @Override
                    public void accept(Dependency a, String s) {
                        id.project().progress.accept(a, s);
                    }
                };
            }

        }

        public boolean cmake(Consumer<String> lineConsumer, List<String> tailopts) {
            List<String> opts = new ArrayList<>();
            opts.add("cmake");
            opts.addAll(tailopts);
            boolean success;
            CMakeProgress cmakeProgres = CMakeProgress.adapt(id());
            cmakeProgres.cmakeCommandLine(this, String.join(" ", opts));
            try {
                var process = new ProcessBuilder()
                        .command(opts)
                        .redirectErrorStream(true)
                        // .redirectOutput(ProcessBuilder.Redirect.INHERIT)
                        .start();
                process.waitFor();
                new BufferedReader(new InputStreamReader(process.getInputStream())).lines()
                        .forEach(line -> {
                            lineConsumer.accept(line);
                            cmakeProgres.cmakeProgress(this, line);
                        });
                success = (process.exitValue() == 0);

                if (!success) {
                    cmakeProgres.cmakeError(this, "ERR " + String.join(" ", opts));
                    throw new RuntimeException("CMake failed");
                }
                cmakeProgres.cmakeInfo(this, "Done " + String.join(" ", opts));
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
            return success;
        }

        @Override
        public List<Path> generatedPaths() {
            throw new IllegalStateException("who called me");
        }

        boolean cmake(Consumer<String> lineConsumer, String... opts) {
            return cmake(lineConsumer, List.of(opts));
        }

        public boolean cmakeInit(Consumer<String> lineConsumer) {
            return cmake(lineConsumer, "--fresh", "-DHAT_TARGET=" + id().project().buildPath(), "-B", cmakeBuildDir().toString(), "-S", cmakeSourceDir().toString());
        }

        public boolean cmakeBuildTarget(Consumer<String> lineConsumer, String target) {
            return cmake(lineConsumer, "--build", cmakeBuildDir().toString(), "--target", target);
        }

        public boolean cmakeBuild(Consumer<String> lineConsumer) {
            return cmake(lineConsumer, "--build", cmakeBuildDir().toString());
        }

        public boolean cmakeClean(Consumer<String> lineConsumer) {
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


    public static class JExtract extends Jar {
        @Override
        public Path javaSourcePath() {
            return id.project.confPath.resolve(id().hyphenatedName).resolve("src/main/java");
        }

        public interface JExtractProgress extends Progress {

            default void jextractProgress(Dependency a, String s) {
                accept(a, "JEXTRACT :" + s);
            }

            default void jextractInfo(Dependency a, String s) {
                accept(a, "JEXTRACT :" + s);
            }

            default void jextractCommandLine(Dependency a, String s) {
                accept(a, "JEXTRACT :" + s);
            }

            default void jextractVerbose(Dependency a, String s) {
                accept(a, "JEXTRACT :     " + s);
            }

            default void jextractError(Dependency a, String s) {
                accept(a, "JEXTRACT :!!     " + s);
            }

            static JExtractProgress adapt(Project.Id id) {
                return (id.project().progress instanceof JExtractProgress progress) ? progress : new JExtractProgress() {
                    @Override
                    public void accept(Dependency a, String s) {
                        id.project().progress.accept(a, s);
                    }
                };
            }
        }


        public interface ExtractSpec {
            Path header();

            List<Path> frameworks();
        }

        record Mac(Path macSdkSysLibFrameWorks, Path macSysLibFrameWorks, Path header,
                   List<Path> frameworks) implements ExtractSpec {
            static Mac of(CMakeInfo cMakeInfo, String... frameworks) {
                var value = (String) cMakeInfo.properties.get("CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES");
                Path macSdkSysLibFrameWorks = Path.of(value);
                Path macSysLibFrameWorks = Path.of("/System/Library/Frameworks");
                var firstName = frameworks[0];
                return new Mac(
                        macSdkSysLibFrameWorks,
                        macSysLibFrameWorks,
                        macSdkSysLibFrameWorks.resolve(firstName.toUpperCase() + ".framework/Headers/" + firstName + ".h"),
                        Stream.of(frameworks).map(s -> macSysLibFrameWorks.resolve(s + ".framework/" + s)).collect(Collectors.toList())
                );
            }

            void writeCompileFlags(Path outputDir) {
                try {
                    Path compileFLags = outputDir.resolve("compile_flags.txt");
                    Files.writeString(compileFLags, "-F" + macSdkSysLibFrameWorks + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE);
                } catch (IOException e) {
                    throw new IllegalStateException(e);
                }
            }
        }

        @Override
        public boolean build() {
            try {
                id.project.mkdir(javaSourcePath());

                List<String> opts = new ArrayList<>(List.of());
                opts.addAll(List.of(
                        "/Users/grfrost/jextract-22/bin/jextract",
                        "--target-package", id().name(),
                        "--output", javaSourcePath().toString()
                ));
                spec.frameworks().forEach(library -> opts.addAll(List.of(
                        "--library", ":" + library
                )));
                opts.addAll(List.of(
                        "--header-class-name", id().name() + "_h",
                        spec.header().toString()
                ));
                if (spec instanceof Mac mac) {
                    mac.writeCompileFlags(id().project().rootPath);
                }
                boolean success;
                JExtractProgress jExtractProgress = JExtractProgress.adapt(id());
                jExtractProgress.jextractCommandLine(this, String.join(" ", opts));
                try {
                    var process = new ProcessBuilder()
                            .command(opts)
                            .redirectErrorStream(true)
                            .start();
                    process.waitFor();
                    new BufferedReader(new InputStreamReader(process.getInputStream())).lines()
                            .forEach(s -> jExtractProgress.jextractProgress(this, s));
                    success = (process.exitValue() == 0);
                    if (!success) {
                        jExtractProgress.jextractError(this, "error " + process.exitValue());
                    }
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
                super.build();
            } catch (Exception e) {
                throw new IllegalStateException(e);
            }
            return false;
        }

        @Override
        public boolean clean() {
            // No opp
            return false;
        }

        final ExtractSpec spec;

        private JExtract(Project.Id id, ExtractSpec spec, Set<Path> exclude, Set<Dependency> dependencies) {
            super(id, exclude, dependencies);
            this.spec = spec;
            id.project.add(this);
        }

        static JExtract of(Project.Id id, ExtractSpec spec, Set<Path> exclude, Set<Dependency> dependencies) {
            return new JExtract(id, spec, exclude, dependencies);
        }

        static JExtract of(Project.Id id, ExtractSpec spec, Set<Path> exclude, Dependency... dependencies) {
            return of(id, spec, exclude, Set.of(dependencies));
        }

        static JExtract of(Project.Id id, ExtractSpec spec, Set<Dependency> dependencies) {
            return new JExtract(id, spec, Set.of(), dependencies);
        }

        static JExtract of(Project.Id id, ExtractSpec spec, Dependency... dependencies) {
            return of(id, spec, Set.of(), Set.of(dependencies));
        }
    }


    public static abstract class CMakeInfo extends Job.CMake implements Job.Dependency.Optional {
       /* interface Mapper{

        }
        interface PathMapper extends Mapper{
            Path map(String value);
            PathMapper impl =  (s)->Path.of(s);
        }
        interface BooleanMapper extends Mapper{
            boolean map(String value);
            BooleanMapper impl =  (s)->Boolean.getBoolean(s);
        }
        interface StringMapper extends Mapper{
            String map(String value);
            StringMapper impl =  (s)->s;
        }*/

        Path asPath(String key) {
            return properties.containsKey(key) ? Path.of((String) properties.get(key)) : null;
        }

        boolean asBoolean(String key) {
            return properties.containsKey(key) && Boolean.parseBoolean((String) properties.get(key));
        }

        String asString(String key) {
            return (properties.containsKey(key) && properties.get(key) instanceof String s) ? s : null;
        }


        final String find;
        final String response;
        final static String template = """
                cmake_minimum_required(VERSION 3.22.1)
                project(extractions)
                find_package(__find__)
                get_cmake_property(_variableNames VARIABLES)
                foreach (_variableName ${_variableNames})
                   message(STATUS "${_variableName}=${${_variableName}}")
                endforeach()
                """;

        final String text;

        final Set<String> vars;
        Properties properties = new Properties();
        final Path propertiesPath;

        final Map<String, String> otherVarMap = new LinkedHashMap<>();
        final boolean available;

        CMakeInfo(Job.Project.Id id, String find, String response, Set<String> vars, Set<Job.Dependency> buildDependencies) {
            super(id, id.project().confPath().resolve("cmake-info").resolve(find), buildDependencies);
            this.find = find;
            this.response = response;
            this.vars = vars;
            this.text = template.replaceAll("__find__", find).replaceAll("__response__", response);
            this.propertiesPath = cmakeSourceDir().resolve("properties");
            if (Files.exists(propertiesPath)) {
                properties = new Properties();
                try {
                    properties.load(Files.newInputStream(propertiesPath));

                } catch (IOException e) {
                    throw new IllegalStateException(e);
                }
            } else {
                id.project().mkdir(cmakeBuildDir());
                try {
                    Files.writeString(CMakeLists_txt, this.text, StandardCharsets.UTF_8, StandardOpenOption.CREATE);
                    Pattern p = Pattern.compile("-- *([A-Za-z_0-9]+)=(.*)");
                    cmakeInit((line) -> {
                        if (p.matcher(line) instanceof Matcher matcher && matcher.matches()) {
                            //   System.out.println("GOT "+matcher.group(1)+"->"+matcher.group(2));
                            if (vars.contains(matcher.group(1))) {
                                properties.put(matcher.group(1), matcher.group(2));
                            } else {
                                otherVarMap.put(matcher.group(1), matcher.group(2));
                            }
                        } else {
                            // System.out.println("skipped " + line);
                        }
                    });
                    properties.store(Files.newOutputStream(propertiesPath), "A comment");
                } catch (IOException ioException) {
                    throw new IllegalStateException(ioException);
                }
            }
            available = asBoolean(response);
        }

        @Override
        public boolean isAvailable() {
            return available;
        }
    }

    public static class Mac extends Job.AbstractArtifact<Mac> implements Job.Dependency.Optional{
        Mac(Job.Project.Id id, Set<Job.Dependency> buildDependencies) {
            super(id,  buildDependencies);
        }

        @Override
        public boolean isAvailable() {
            return System.getProperty("os.name").toLowerCase().contains("mac");
        }
    }
    public static class Linux extends Job.AbstractArtifact<Linux> implements Job.Dependency.Optional{
        Linux(Job.Project.Id id, Set<Job.Dependency> buildDependencies) {
            super(id,  buildDependencies);
        }

        @Override
        public boolean isAvailable() {
            return System.getProperty("os.name").toLowerCase().contains("linux");
        }
    }
    public static class OpenGL extends Job.CMakeInfo {

        final Path glLibrary;
        OpenGL(Job.Project.Id id, Set<Job.Dependency> buildDependencies) {
            super(id,  "OpenGL", "OPENGL_FOUND",Set.of(
                    "OPENGL_FOUND",
                    "OPENGL_GLU_FOUND",
                    "OPENGL_gl_LIBRARY",
                    "OPENGL_glu_LIBRARY",
                    "CMAKE_HOST_SYSTEM_NAME",
                    "CMAKE_HOST_SYSTEM_PROCESSOR",
                    "CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES"
            ), buildDependencies);
            System.out.println("HERE");
            glLibrary = asPath("OpenGL_glu_Library");
        }

    }

   public static class OpenCL extends Job.CMakeInfo{


        OpenCL(Job.Project.Id id, Set<Job.Dependency> buildDependencies) {
            super(id,  "OpenCL", "OPENCL_FOUND", Set.of(
                    "OPENCL_FOUND",
                    "CMAKE_HOST_SYSTEM_NAME",
                    "CMAKE_HOST_SYSTEM_PROCESSOR",
                    "CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES"
            ), buildDependencies);

        }
    }
  public static   class Cuda extends Job.CMakeInfo{
        Cuda(Job.Project.Id id, Set<Job.Dependency> buildDependencies) {
            super(id,  "CUDAToolkit", "CUDATOOLKIT_FOUND",Set.of(
                    "CUDATOOLKIT_FOUND",
                    "CMAKE_HOST_SYSTEM_NAME",
                    "CMAKE_HOST_SYSTEM_PROCESSOR",
                    "CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES"

            ), buildDependencies);
        }
    }
}



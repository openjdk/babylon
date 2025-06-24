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

import javax.lang.model.element.Modifier;
import javax.lang.model.element.NestingKind;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.ToolProvider;

import static java.lang.IO.println;

public static String separated(List<String> strings, String separator) {
    StringBuilder stringBuilder = new StringBuilder();
    strings.forEach(opt -> {
        if (opt != null) {
            stringBuilder.append(stringBuilder.isEmpty() ? "" : separator).append(opt);
        }
    });
    return stringBuilder.toString();
}

static boolean process(Consumer<String> consumer, List<String> opts) {
    if (consumer == null) {
        consumer = (s) -> println("NULL consumer" + s);
    }
    boolean success = false;
    try {
        var process = new ProcessBuilder().command(opts).redirectErrorStream(true).start();
        process.waitFor();
        new BufferedReader(new InputStreamReader(process.getInputStream())).lines().forEach(consumer);
        success = (process.exitValue() == 0);
        if (!success) {
            println("process returned error " + process.exitValue());
        }
    } catch (Exception e) {
        e.printStackTrace();
        throw new IllegalStateException(e);
    }
    return success;
}

static boolean process(List<String> opts) {
    return process(s -> println(s), opts);
}

static boolean process(Consumer<String> consumer, String... opts) {
    return process(consumer, List.of(opts));
}

static boolean process(String... opts) {
    return process(s -> println(s), List.of(opts));
}

public interface ProjectProvider {
    Project project();

    default void run(Consumer<String> consumer, ProjectProvider backend, List<String> args) throws IOException, InterruptedException {
        project().runit(project().mainClassName, backend.project(), consumer, args);
    }
    default void run(Consumer<String> consumer, ProjectProvider backend, String ... args) throws IOException, InterruptedException {
       run(consumer, backend.project(), List.of(args));
    }

    default void run(Consumer<String> consumer, List<String> args) throws IOException, InterruptedException {
        project().runit(project().mainClassName, null, consumer, args);
    }
    default void run(Consumer<String> consumer, String args) throws IOException, InterruptedException {
       run (consumer, List.of(args));
    }

    default ProjectProvider compile(Consumer<String> consumer) {
        if (project().needsCompiling()) {
            return project().compileIt(consumer);
        }
        return this;
    }
}
public static final class JavaSource extends SimpleJavaFileObject {
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
    static JavaSource of(Path path) {
        return new JavaSource(path);
    }

    public Path path() {
        return path;
    }
}
public record Project(String name, String mainClassName, Path root, Path sourcePath, Path buildDir, Path classDir,
                      Set<ProjectProvider> dependencies, Path jarFile,
                      List<JavaSource> sourceFiles) implements ProjectProvider {

    public record JavaSourceN(Path path, SimpleJavaFileObject impl) implements JavaFileObject {
        @Override
        public URI toUri() {
            return impl.toUri();
        }

        @Override
        public String getName() {
            return impl.getName();
        }

        @Override
        public InputStream openInputStream() throws IOException {
            return impl.openInputStream();
        }

        @Override
        public OutputStream openOutputStream() throws IOException {
            return impl.openOutputStream();
        }

        @Override
        public Reader openReader(boolean ignoreEncodingErrors) throws IOException {
            return impl.openReader(ignoreEncodingErrors);
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            try {
                return Files.readString(Path.of(toUri()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public Writer openWriter() throws IOException {
            return impl.openWriter();
        }

        @Override
        public long getLastModified() {
            return impl.getLastModified();
        }

        @Override
        public boolean delete() {
            return impl.delete();
        }

        static JavaSourceN of(Path path) {
            return new JavaSourceN(path,  new SimpleJavaFileObject(path.toUri(), Kind.SOURCE){});
        }

        @Override
        public Kind getKind() {
            return impl.getKind();
        }

        @Override
        public boolean isNameCompatible(String simpleName, Kind kind) {
            return impl.isNameCompatible(simpleName, kind);
        }

        @Override
        public NestingKind getNestingKind() {
            return impl.getNestingKind();
        }

        @Override
        public Modifier getAccessLevel() {
            return impl.getAccessLevel();
        }
    }


    public Path path() {
        return jarFile;
    }

    @Override
    public Project project() {
        return this;
    }

    public boolean needsCompiling() {
        return !Files.exists(jarFile);
    }

    public record Result(Project project, boolean ok, List<JavaFileObject> classes,
                         Path jar,
                         List<PathAndRoot> jaredPathsWithRoots, List<Path> jarPaths,
                         List<Diagnostic<? extends JavaFileObject>> javacDiagnostics,
                         List<Diagnostic<?>> jarDiagnostics) implements ProjectProvider { }

    public static Project of(String name, String mainClassName, Path root, String version, Path buildDir, ProjectProvider... classPathEntries) {
        try {
            var sourcePath = root.resolve("src/main/java");
            var src = Files.walk(sourcePath).filter(Files::isRegularFile).filter(path -> path.toString().endsWith(".java")).map(JavaSource::of).toList();
            var classDir = buildDir.resolve(name + "-" + version + ".classes");
            var jarFile = buildDir.resolve(name + "-" + version + ".jar");
            return new Project(name, mainClassName, root, sourcePath, buildDir, classDir, new LinkedHashSet<>(Arrays.asList(classPathEntries)), jarFile, src);
        } catch (IOException e) {
            // throw new RuntimeException(e);
            return null;
        }
    }


    public static Project example(Path root, Path buildDir, ProjectProvider... classPathEntries) {
        return of("hat-example-" + root.getFileName().toString(), root.getFileName().toString() + ".Main", root, "1.0", buildDir, classPathEntries);
    }

    public static Project backend(Path root, Path buildDir, ProjectProvider... classPathEntries) {

        return of("hat-backend-" + root.getParent().getFileName().toString() + "-" + root.getFileName().toString(), null, root, "1.0", buildDir, classPathEntries);
    }

    Result compileIt(Consumer<String> consumer) {
        if (!needsCompiling()) {
            throw new IllegalStateException("does not need compiling not called");
        }

        List<String> opts = new ArrayList<>(
                List.of(
                        "--source", "26",
                        "--enable-preview",
                        "--add-modules", "jdk.incubator.code",
                        "-d", classDir.toString(),
                        "--class-path", separated(dependencies.stream().map(cpe -> cpe.project().jarFile.toString()).toList(), ":"),
                        "--source-path", sourcePath.toString()
                )
        );
        //   if (verbose) {
        println(separated(opts, " "));
        // }
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        DiagnosticCollector<JavaFileObject> javacDiagnostics = new DiagnosticCollector<>();


        JavaCompiler.CompilationTask compilationTask =
                (javac.getTask(
                        new PrintWriter(System.err),
                        javac.getStandardFileManager(javacDiagnostics, null, null),
                        javacDiagnostics,
                        opts,
                        null,
                        sourceFiles
                ));
        JavacTask javacTask = (JavacTask) compilationTask;
        List<JavaFileObject> generatedClasses = new ArrayList<>();
        try {
            rmdir(classDir);
            Files.createDirectories(classDir);

            javacTask.generate().forEach(generatedClasses::add);
            Path resourceDir = root.resolve("src/main/resources");

            List<Path> dirsToJar = new ArrayList<>(List.of(classDir));
            if (Files.exists(resourceDir)) {
                dirsToJar.add(resourceDir);
            }
            List<Diagnostic<?>> jarDiagnostics = new ArrayList<>();
            var jarStream = new JarOutputStream(Files.newOutputStream(jarFile));
            Manifest manifest = null;//new Manifest(null, null, null,null,null);
            if (manifest != null) {
                var entry = new JarEntry("META-INF/MANIFEST.MF");
                jarStream.putNextEntry(entry);
                manifest.writeTo(jarStream);
                jarStream.closeEntry();
            }


            List<PathAndRoot> pathsToJar = new ArrayList<>();
            dirsToJar.forEach(root -> {
                try {
                    Files.walk(root).map(path -> new PathAndRoot(root, path)).forEach(pathsToJar::add);
                } catch (Exception e) {

                }
            });
            List<Path> filePaths = new ArrayList<>();
            pathsToJar.stream()
                    .sorted(Comparator.comparing(PathAndRoot::path))
                    .forEach(
                            rootAndPath -> {
                                try {
                                    if (!Files.isDirectory(rootAndPath.path)) {
                                        filePaths.add(rootAndPath.path);
                                        var relative = rootAndPath.root.relativize(rootAndPath.path);
                                        var entry = new JarEntry(relative.toString());
                                        entry.setTime(Files.getLastModifiedTime(rootAndPath.path()).toMillis());
                                        jarStream.putNextEntry(entry);
                                        Files.newInputStream(rootAndPath.path()).transferTo(jarStream);
                                        jarStream.closeEntry();
                                        //   if (verbose) {
                                        //println("INFO: adding " + rootAndPath.relativize().toString());
                                        //  }
                                    }
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            });
            jarStream.finish();
            jarStream.close();
            // if (verbose) {
            // println("INFO: created " + jarFile.toString());
            //}
            return new Result(this, true, generatedClasses, jarFile, pathsToJar, filePaths, javacDiagnostics.getDiagnostics(), jarDiagnostics);
        } catch (IOException e) {
            println(e.getMessage());
            return null;
        }
    }

    public record PathAndRoot(Path root, Path path) {
        Path relativize() {
            return root.relativize(path());
        }
    }

    public record Manifest(String mainClass, String[] classPath, String version, String createdBy, String buildBy) {

        public void writeTo(JarOutputStream jarStream) {
            PrintWriter printWriter = new PrintWriter(jarStream);
            if (version != null) {
                printWriter.println("Manifest-Version: " + version);
            }
            if (mainClass != null) {
                printWriter.println("Main-Class: " + mainClass);
            }
            if (classPath != null) {
                printWriter.print("Class-Path:");
                for (String s : classPath) {
                    printWriter.print(" ");
                    printWriter.print(s);
                }
                printWriter.println();
            }
            printWriter.flush();
        }
    }


    void runit(String classToRun, ProjectProvider backend, Consumer<String> consumer, List<String> args) throws IOException, InterruptedException {
        // before we start lets ensure all the dependencies are compiled
        Set<ProjectProvider> allDependencies = new LinkedHashSet<>();
        dependencies.forEach(classPathEntry -> classPathEntry.project().collect(allDependencies));
        if (backend != null) {
            backend.project().collect(allDependencies);
        }
        allDependencies.add(this);
        allDependencies.forEach(p ->
                p.compile(consumer)
        );
        List<String> opts = new ArrayList<>();

        var colonSeperated = separated(allDependencies.stream().filter(Objects::nonNull).map(cpe -> cpe.project().jarFile.toString()).toList(), ":");

        opts.addAll(List.of(
                "/Users/grfrost/github/babylon-grfrost-fork/build/macosx-aarch64-server-release/jdk/bin/java",
                "--enable-preview",
                "--enable-native-access=ALL-UNNAMED",
                "--class-path", colonSeperated,
                "-Djava.library.path=" + buildDir.toString(),
                classToRun != null ? classToRun : mainClassName
        ));
        opts.addAll(args);
        process(consumer, opts);
    }

    private void collect(Set<ProjectProvider> all) {

        dependencies.forEach(classPathEntry -> classPathEntry.project().collect(all));
        all.add(this);
    }

}

static Path rmdir(Path path) {
    try {
        if (Files.exists(path)) {
            Files.walk(path)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
    } catch (IOException ioe) {
        System.out.println(ioe);
        throw new RuntimeException(ioe);
    }
    return path;
}

record CMaker(boolean verbose, Path dir, Path hatBuildDir, Path cmakeBuildDir) {

    static CMaker of(boolean verbose, Path dir, Path hatBuildDir) {
        var cmakeBuildDir = dir.resolve("build");
        try {
            rmdir(cmakeBuildDir);
            Files.createDirectories(cmakeBuildDir);
            return new CMaker(verbose, dir, hatBuildDir, cmakeBuildDir);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public boolean init(Consumer<String> consumer) {
        return process(consumer, "cmake",  "--fresh", "-DHAT_TARGET=" + hatBuildDir, "-B", cmakeBuildDir.toString(), "-S", dir.toString());
    }

    public boolean build(Consumer<String> consumer) {
        return process(consumer, "cmake", "--build", cmakeBuildDir.toString());
    }
    public boolean build(Consumer<String> consumer, String target ) {
        return process(consumer, "cmake", "--build", cmakeBuildDir.toString(), "--target", target);
    }
}


public static final Path userDir = Path.of(System.getProperty("user.dir"));
public static final Path rootDir = userDir.getFileName().toString().equals("intellij") ? userDir.getParent() : userDir;
public static final Path buildDir = rootDir.resolve("build");
public static final Project hatCore = Project.of("hat-core", null, rootDir.resolve("hat-core"), "1.0", buildDir);
public static final Path backendsDir = rootDir.resolve("backends");
public static final Path ffiBackendsDir = backendsDir.resolve("ffi");
public static final Project ffiSharedBackend = Project.backend(ffiBackendsDir.resolve("shared"), buildDir, hatCore);
public static final Project cudaFFiBackend = Project.backend(ffiBackendsDir.resolve("cuda"), buildDir, hatCore, ffiSharedBackend);
public static final Project openclFFiBackend = Project.backend(ffiBackendsDir.resolve("opencl"), buildDir, hatCore, ffiSharedBackend);
public static final Project mockFFiBackend = Project.backend(ffiBackendsDir.resolve("mock"), buildDir, hatCore, ffiSharedBackend);
public static final Path javaBackendsDir = backendsDir.resolve("java");
public static final Project javaMtBackend = Project.backend(javaBackendsDir.resolve("mt"), buildDir, hatCore);
public static final Project javaSeqBackend = Project.backend(javaBackendsDir.resolve("seq"), buildDir, hatCore);
public static final Path examplesDir = rootDir.resolve("examples");
public static final Project mandel = Project.example(examplesDir.resolve("mandel"), buildDir, hatCore);
public static final Project life = Project.example(examplesDir.resolve("life"), buildDir, hatCore);
public static final Project squares = Project.example(examplesDir.resolve("squares"), buildDir, hatCore);
public static final Project heal = Project.example(examplesDir.resolve("heal"), buildDir, hatCore);
public static final Project violaJones = Project.example(examplesDir.resolve("violajones"), buildDir, hatCore);
public static final List<Project> examples = List.of(squares,mandel,life,heal,violaJones);
public static final List<Project> backends = List.of(openclFFiBackend,javaMtBackend,javaSeqBackend,cudaFFiBackend,mockFFiBackend);
public static Project example(String n){
    return examples.stream()
            //.peek(p->println(p.name()))
            .filter(p->p.name().equals("hat-example-"+n))
            .findFirst().orElse(null);
}
public static Project backend(String n){
    return backends.stream()
            .peek(p->println(p.name()))
            .filter(p->p.name().equals("hat-backend-"+n))
            .findFirst().orElse(null);
}




public static void main(String[] argArr) throws IOException, InterruptedException {

    var args = new ArrayList<>(List.of(argArr));
    var out = (Consumer<String>) IO::println;
    var command = args.isEmpty()?null:args.removeFirst();

    if ("clean".equals(command)) {
        rmdir(buildDir);

        command = args.isEmpty()?null:args.removeFirst();
        if (command == null) {
            System.exit(0);
        }
    }
    if ("bld".equals(command)) {
        backends.forEach(p->((ProjectProvider)p).compile(out));
        examples.forEach(p ->((ProjectProvider)p).compile(out) );
        if (CMaker.of(false, ffiBackendsDir, buildDir) instanceof CMaker cmaker) {
            if (cmaker.init(out)) {
                if (cmaker.build(out)) {
                    System.out.println("cmake finished");
                }
            }
        }
    }else if ("run".equals(command) && args.size()>1){
        var backendName =args.removeFirst();
        var exampleName = args.removeFirst();
        System.out.println("Try to run example "+exampleName+" with backend "+backendName);
        if (backend(backendName) instanceof ProjectProvider backend){
            if (backendName.startsWith("ffi") && !Files.exists(backend.project().jarFile)) {
                if (CMaker.of(false, ffiBackendsDir, buildDir) instanceof CMaker cmaker) {
                    if (cmaker.init(out)) {
                        if (cmaker.build(out, backendName.substring(4)+"_backend")) {
                            System.out.println("cmake created "+backendName+" backend");
                        }
                    }
                }
            }
            if (example(exampleName) instanceof ProjectProvider example){
                example.run(out, backend, args);
            }else{
                System. out.println("Failed to find "+exampleName);
            }
        }else{
            System. out.println("Failed to find backend "+backendName);
        }

     }else{
         System. out.println("What? ");
     }

}



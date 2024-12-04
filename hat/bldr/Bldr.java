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

package bldr;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipFile;

import static java.io.IO.print;
import static java.io.IO.println;
import static java.nio.file.Files.isDirectory;
import static java.nio.file.Files.isRegularFile;

public class Bldr {
    public sealed interface PathHolder permits ClassPathEntry, DirPathHolder, Executable, FilePathHolder {
        Path path();

        default String fileName() {
            return path().getFileName().toString();
        }

        default Matcher pathMatcher(Pattern pattern) {
            return pattern.matcher(path().toString());
        }

        default boolean matches(Pattern pattern) {
            return pathMatcher(pattern).matches();
        }

        default boolean matches(String pattern) {
            return pathMatcher(Pattern.compile(pattern)).matches();
        }

        default CppSourceFile cppSourceFile(String s) {
            return CppSourceFile.of(path().resolve(s));
        }

        default XMLFile xmlFile(String s) {
            return XMLFile.of(path().resolve(s));
        }

        default TestNGSuiteFile testNGSuiteFile(String s) {
            return TestNGSuiteFile.of(path().resolve(s));
        }
    }

    public sealed interface DirPathHolder<T extends DirPathHolder<T>> extends PathHolder
            permits BuildDirHolder, Dir, SourcePathEntry {
        default Path path(String subdir) {
            return path().resolve(subdir);
        }

        default Stream<Path> find() {
            try {
                return Files.walk(path());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        default Stream<Path> find(Predicate<Path> predicate) {
            return find().filter(predicate);
        }

        default Stream<Path> findFiles() {
            return find(Files::isRegularFile);
        }

        default Stream<Path> findDirs() {
            return find(Files::isDirectory);
        }

        default Stream<Path> findFiles(Predicate<Path> predicate) {
            return findFiles().filter(predicate);
        }

        default Stream<Path> findFilesBySuffix(String suffix) {
            return findFiles(p -> p.toString().endsWith(suffix));
        }

        default Stream<SearchableTextFile> findTextFiles(String... suffixes) {
            return findFiles()
                    .map(SearchableTextFile::new)
                    .filter(searchableTextFile -> searchableTextFile.hasSuffix(suffixes));
        }

        default Stream<Path> findDirs(Predicate<Path> predicate) {
            return find(Files::isDirectory).filter(predicate);
        }

        default boolean exists() {
            return Files.exists(path()) && Files.isDirectory(path());
        }

        default BuildDir buildDir(String name) {
            return BuildDir.of(path().resolve(name));
        }

        default SourcePathEntry sourceDir(String s) {
            return SourcePathEntry.of(path().resolve(s));
        }
    }

    public sealed interface FilePathHolder extends PathHolder {
        default boolean exists() {
            return Files.exists(path()) && Files.isRegularFile(path());
        }
    }

    public sealed interface Executable extends PathHolder {
        default boolean exists() {
            return Files.exists(path()) && Files.isRegularFile(path()) && Files.isExecutable(path());
        }
    }


    public interface ClassPathEntryProvider {
        List<Bldr.ClassPathEntry> classPathEntries();
    }

    public sealed interface ClassPathEntry extends PathHolder, ClassPathEntryProvider {
    }

    interface PathHolderList<T extends PathHolder> {
        List<T> entries();

        default String charSeparated() {
            StringBuilder sb = new StringBuilder();
            entries().forEach(pathHolder -> {
                                if (!sb.isEmpty()) {
                                    sb.append(File.pathSeparatorChar);
                                }
                                sb.append(pathHolder.path());
                            });
            return sb.toString();
        }
    }

    public record ClassPath(List<ClassPathEntry> classPathEntries)
            implements PathHolderList<ClassPathEntry>, ClassPathEntryProvider {
        public static ClassPath of() {
            return new ClassPath(new ArrayList<>());
        }

        public static ClassPath ofOrUse(ClassPath classPath) {
            return classPath == null ? of() : classPath;
        }

        public ClassPath add(List<ClassPathEntryProvider> classPathEntryProviders) {
            classPathEntryProviders.forEach(
                    classPathEntryProvider ->
                            this.classPathEntries.addAll(classPathEntryProvider.classPathEntries()));
            return this;
        }

        public ClassPath add(ClassPathEntryProvider... classPathEntryProviders) {
            return add(List.of(classPathEntryProviders));
        }

        @Override
        public String toString() {
            return charSeparated();
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return this.classPathEntries;
        }

        @Override
        public List<ClassPathEntry> entries() {
            return this.classPathEntries;
        }
    }

    public record SourcePath(List<SourcePathEntry> entries)
            implements PathHolderList<SourcePathEntry> {
        public static SourcePath of() {
            return new SourcePath(new ArrayList<>());
        }

        public static SourcePath ofOrUse(SourcePath sourcePath) {
            return sourcePath == null ? of() : sourcePath;
        }

        public SourcePath add(List<SourcePathEntry> sourcePathEntries) {
            entries.addAll(sourcePathEntries);
            return this;
        }

        public SourcePath add(SourcePathEntry... sourcePathEntries) {
            add(Arrays.asList(sourcePathEntries));
            return this;
        }

        public SourcePath add(SourcePath... sourcePaths) {
            List.of(sourcePaths).forEach(sourcePath -> add(sourcePath.entries));
            return this;
        }

        @Override
        public String toString() {
            return charSeparated();
        }

        public Stream<Path> javaFiles() {
            List<Path> javaFiles = new ArrayList<>();
            entries.forEach(entry -> entry.javaFiles().forEach(javaFiles::add));
            return javaFiles.stream();
        }
    }

    public record DirPath(List<DirPathHolder<?>> entries)
            implements PathHolderList<DirPathHolder<?>> {
        public static DirPath of() {
            return new DirPath(new ArrayList<>());
        }

        public static DirPath ofOrUse(DirPath dirPath) {
            return dirPath == null ? of() : dirPath;
        }

        public DirPath add(List<DirPathHolder<?>> dirPathHolders) {
            entries.addAll(dirPathHolders);
            return this;
        }

        public DirPath add(DirPathHolder<?>... dirPathHolders) {
            add(Arrays.asList(dirPathHolders));
            return this;
        }

        public DirPath add(DirPath... dirPaths) {
            List.of(dirPaths).forEach(dirPath -> add(dirPath.entries));
            return this;
        }

        @Override
        public String toString() {
            return charSeparated();
        }
    }

    public record CMakeBuildDir(Path path) implements BuildDirHolder<CMakeBuildDir> {
        public static CMakeBuildDir of(Path path) {
            return new CMakeBuildDir(path);
        }

        @Override
        public CMakeBuildDir create() {
            return CMakeBuildDir.of(mkdir(path()));
        }

        @Override
        public CMakeBuildDir remove() {
            return CMakeBuildDir.of(rmdir(path()));
        }
    }

    public sealed interface BuildDirHolder<T extends BuildDirHolder<T>> extends DirPathHolder<T> {
        T create();

        T remove();

        default void clean() {
            remove();
            create();
        }

        default Path mkdir(Path path) {
            try {
                return Files.createDirectories(path);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        default Path rmdir(Path path) {
            try {
                if (Files.exists(path)) {
                    Files.walk(path)
                            .sorted(Comparator.reverseOrder())
                            .map(Path::toFile)
                            .forEach(File::delete);
                }
            } catch (IOException ioe) {
                System.out.println(ioe);
            }
            return path;
        }
    }

    public record ClassDir(Path path) implements ClassPathEntry, BuildDirHolder<ClassDir> {
        public static ClassDir of(Path path) {
            return new ClassDir(path);
        }

        public static ClassDir temp() {
            try {
                return of(Files.createTempDirectory("javacClasses"));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public ClassDir create() {
            return ClassDir.of(mkdir(path()));
        }

        @Override
        public ClassDir remove() {
            return ClassDir.of(rmdir(path()));
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(this);
        }
    }

    public record RepoDir(Path path) implements BuildDirHolder<RepoDir> {
        public static RepoDir of(Path path) {
            return new RepoDir(path);
        }

        @Override
        public RepoDir create() {
            return RepoDir.of(mkdir(path()));
        }

        @Override
        public RepoDir remove() {
            return RepoDir.of(rmdir(path()));
        }

        public JarFile jarFile(String name) {
            return JarFile.of(path().resolve(name));
        }

        public ClassPathEntryProvider classPathEntries(String... specs) {
            var repo = new MavenStyleRepository(this);
            return repo.classPathEntries(specs);
        }
    }

    public record Dir(Path path) implements DirPathHolder<Dir> {
        public static Dir of(Path path) {
            return new Dir(path);
        }

        public static Dir of(String string) {
            return of(Path.of(string));
        }

        public static Dir ofExisting(String string) {
            return of(assertExists(Path.of(string)));
        }

        public static Dir current() {
            return of(Path.of(System.getProperty("user.dir")));
        }

        public Dir parent() {
            return of(path().getParent());
        }

        public Dir dir(String subdir) {
            return Dir.of(path(subdir));
        }

        public Dir existingDir(String subdir) {
            return assertExists(Dir.of(path(subdir)));
        }

        public Stream<Dir> subDirs() {
            return Stream.of(Objects.requireNonNull(path().toFile().listFiles(File::isDirectory)))
                    .map(d -> Dir.of(d.getPath()));
        }

        public Stream<Dir> subDirs(Predicate<Dir> predicate) {
            return subDirs().filter(predicate);
        }

        public Dir forEachSubDir(Predicate<Dir> predicate, Consumer<Dir> consumer) {
            subDirs().filter(predicate).forEach(consumer);
            return this;
        }



        public XMLFile pom(
                String comment, Consumer<XMLNode.PomXmlBuilder> pomXmlBuilderConsumer) {
            XMLFile xmlFile = xmlFile("pom.xml");
            XMLNode.createPom(comment, pomXmlBuilderConsumer).write(xmlFile);
            return xmlFile;
        }

        public BuildDir existingBuildDir(String subdir) {
            return assertExists(BuildDir.of(path(subdir)));
        }
    }

    public record SourcePathEntry(Path path) implements DirPathHolder<SourcePathEntry> {
        public static SourcePathEntry of(Path path) {
            return new SourcePathEntry(path);
        }

        public Stream<Path> javaFiles() {
            return findFilesBySuffix(".java");
        }
    }

    public record RootDirAndSubPath(DirPathHolder<?> root, Path path) {
        Path relativize() {
            return root().path().relativize(path());
        }
    }

    public record BuildDir(Path path) implements ClassPathEntry, BuildDirHolder<BuildDir> {
        public static BuildDir of(Path path) {
            return new BuildDir(path);
        }

        public JarFile jarFile(String name) {
            return JarFile.of(path().resolve(name));
        }
        public ClassPathEntryProvider jarFiles(String ...names) {
            var classPath = ClassPath.of();
            Stream.of(names).forEach(name->
                classPath.add(JarFile.of(path().resolve(name))
                )
            );
            return classPath;
        }


        public JarFile jarFile(String name, BiConsumer<JarBuilder, JarFile> biConsumer) {
            var result = JarFile.of(path().resolve(name));
            return result.create(biConsumer);
        }

        public JarFile jarFile(String name, Consumer<JarBuilder> consumer) {
            var result = JarFile.of(path().resolve(name));
            return result.create(consumer);
        }

        public CMakeBuildDir cMakeBuildDir(String name) {
            return CMakeBuildDir.of(path().resolve(name));
        }

        public ClassDir classDir(String name) {
            return ClassDir.of(path().resolve(name));
        }

        public RepoDir repoDir(String name) {
            return RepoDir.of(path().resolve(name));
        }

        @Override
        public BuildDir create() {
            return BuildDir.of(mkdir(path()));
        }

        @Override
        public BuildDir remove() {
            return BuildDir.of(rmdir(path()));
        }

        public BuildDir dir(String subdir) {
            return BuildDir.of(path(subdir));
        }

        public ObjectFile objectFile(String name) {
            return ObjectFile.of(path().resolve(name));
        }

        public ExecutableFile executableFile(String name) {
            return ExecutableFile.of(path().resolve(name));
        }

        public SharedLibraryFile sharedLibraryFile(String name) {
            return SharedLibraryFile.of(path().resolve(name));
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(this);
        }

        public SearchableTextFile textFile(String file, List<String> strings) {
            SearchableTextFile textFile = SearchableTextFile.of(path().resolve(file));
            try {
                PrintWriter pw = new PrintWriter(Files.newOutputStream(textFile.path(), StandardOpenOption.CREATE));
                strings.forEach(pw::print);
                pw.close();
                return textFile;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public record JarFile(Path path) implements ClassPathEntry, FilePathHolder {
        public static JarFile of(Path path) {
            return new JarFile(path);
        }

        public JarFile create(BiConsumer<JarBuilder, JarFile> jarBuilderConsumer) {
            JarBuilder jarBuilder = new JarBuilder();
            jarBuilder.jar(this);
            jarBuilderConsumer.accept(jarBuilder, this);
            return jar(jarBuilder);
        }

        public JarFile create(Consumer<JarBuilder> jarBuilderConsumer) {
            JarBuilder jarBuilder = new JarBuilder();
            jarBuilder.jar(this);
            jarBuilderConsumer.accept(jarBuilder);
            return jar(jarBuilder);
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(this);
        }
    }

    public sealed interface TextFile extends FilePathHolder {

        static Path tempContaining(String suffix, String text) {
            try {
                var path = Files.createTempFile(Files.createTempDirectory("bldr"), "data", suffix);
                Files.newOutputStream(path).write(text.getBytes());
                return path;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public sealed interface SourceFile extends TextFile {
    }

    public static final class JavaSourceFile extends SimpleJavaFileObject implements SourceFile {
        Path path;

        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            try {
                return Files.readString(Path.of(toUri()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        JavaSourceFile(Path path) {
            super(path.toUri(), JavaFileObject.Kind.SOURCE);
        }

        @Override
        public Path path() {
            return path;
        }
    }

    public record Jextract(Path path) implements Executable {
        public static Jextract of(Path path) {
            return new Jextract(path);
        }

        public void extract(Consumer<JExtractBuilder> jextractBuilderConsumer) {
            jextract(this, jextractBuilderConsumer);
        }
    }


    public record ObjectFile(Path path) implements FilePathHolder {
        public static ObjectFile of(Path path) {
            return new ObjectFile(path);
        }
    }

    public record ExecutableFile(Path path) implements FilePathHolder {
        public static ExecutableFile of(Path path) {
            return new ExecutableFile(path);
        }
    }

    public record SharedLibraryFile(Path path) implements FilePathHolder {
        public static SharedLibraryFile of(Path path) {
            return new SharedLibraryFile(path);
        }
    }

    public record CppSourceFile(Path path) implements SourceFile {
        public static CppSourceFile of(Path path) {
            return new CppSourceFile(path);
        }
    }

    public record CppHeaderSourceFile(Path path) implements SourceFile {
    }

    public record XMLFile(Path path) implements TextFile {
        public static XMLFile of(Path path) {
            return new XMLFile(path);
        }

        public static XMLFile containing(String text) {
            return XMLFile.of(TextFile.tempContaining("xml", text));
        }
    }

    public record TestNGSuiteFile(Path path) implements TextFile {
        public static TestNGSuiteFile of(Path path) {
            return new TestNGSuiteFile(path);
        }

        public static TestNGSuiteFile containing(String text) {
            return TestNGSuiteFile.of(TextFile.tempContaining("xml", text));
        }
    }

    public interface OS {
        String arch();

        String name();

        String version();

        String MacName = "Mac OS X";
        String LinuxName = "Linux";

        record Linux(String arch, String name, String version) implements OS {
        }

        record Mac(String arch, String name, String version) implements OS {
            public Path appLibFrameworks() {
                return Path.of(
                        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
                                + "MacOSX.sdk/System/Library/Frameworks");
            }

            public Path frameworkHeader(String frameworkName, String headerFileName) {
                return appLibFrameworks().resolve(frameworkName + ".framework/Headers/" + headerFileName);
            }

            public Path libFrameworks() {
                return Path.of("/System/Library/Frameworks");
            }

            public Path frameworkLibrary(String frameworkName) {
                return libFrameworks().resolve(frameworkName + ".framework/" + frameworkName);
            }
        }

        static OS get() {
            String arch = System.getProperty("os.arch");
            String name = System.getProperty("os.name");
            String version = System.getProperty("os.version");
            return switch (name) {
                case "Mac OS X" -> new Mac(arch, name, version);
                case "Linux" -> new Linux(arch, name, version);
                default -> throw new IllegalStateException("No os mapping for " + name);
            };
        }
    }

    public static OS os = OS.get();

    public record Java(String version, Dir home) {
    }

    public static Java java =
            new Java(System.getProperty("java.version"), Dir.of(System.getProperty("java.home")));

    public record User(Dir home, Dir pwd) {
    }

    public static User user =
            new User(Dir.of(System.getProperty("user.home")), Dir.of(System.getProperty("user.dir")));

    public abstract static class Builder<T extends Builder<T>> {
        @SuppressWarnings("unchecked")
        T self() {
            return (T) this;
        }

        public boolean verbose;

        public T verbose(boolean verbose) {
            this.verbose = verbose;
            return self();
        }

        public T verbose() {
            verbose(true);
            return self();
        }

        public T when(boolean condition, Consumer<T> consumer) {
            if (condition) {
                consumer.accept(self());
            }
            return self();
        }

        public <P extends PathHolder> T whenExists(P pathHolder, Consumer<T> consumer) {
            if (Files.exists(pathHolder.path())) {
                consumer.accept(self());
            }
            return self();
        }

        public <P extends PathHolder> T whenExists(P pathHolder, BiConsumer<P, T> consumer) {
            if (Files.exists(pathHolder.path())) {
                consumer.accept(pathHolder, self());
            }
            return self();
        }

        public T either(boolean condition, Consumer<T> trueConsumer, Consumer<T> falseConsumer) {
            if (condition) {
                trueConsumer.accept(self());
            } else {
                falseConsumer.accept(self());
            }
            return self();
        }
    }

    public abstract static class OptsBuilder<T extends OptsBuilder<T>> extends Builder<T> {

        public List<String> opts = new ArrayList<>();

        public T opts(List<String> opts) {
            this.opts.addAll(opts);
            return self();
        }

        public T opts(String... opts) {
            opts(Arrays.asList(opts));
            return self();
        }

    }

    public static class JavaOpts<T extends JavaOpts<T>> extends OptsBuilder<T> {
        public Dir jdk = java.home;

        public T opts(JavaOpts<?> javaOpts) {
            return opts(javaOpts.opts);
        }

        static public JavaOpts<?> of() {
            return new JavaOpts<>();
        }

        public T jdk(Dir jdk) {
            this.jdk = jdk;
            return self();
        }

        public T add_exports(String fromModule, String pack, String toModule) {
            return opts("--add-exports=" + fromModule + "/" + pack + "=" + toModule);
        }

        public T add_modules(String... modules) {
            List.of(modules).forEach(module -> opts("--add-modules=" + module));
            return self();
        }

        public T add_exports(String fromModule, List<String> packages, String toModule) {
            packages.forEach(p -> add_exports(fromModule, p, toModule));
            return self();
        }

        public T enable_preview() {
            return opts("--enable-preview");
        }

    }

    public abstract static class JavaToolBuilder<T extends JavaToolBuilder<T>> extends JavaOpts<T> {
        public ClassPath classPath;

        public T class_path(List<ClassPathEntryProvider> classPathEntryProviders) {
            this.classPath = ClassPath.ofOrUse(this.classPath).add(classPathEntryProviders);
            return self();
        }

        public T class_path(ClassPathEntryProvider... classPathEntryProviders) {
            return class_path(List.of(classPathEntryProviders));
        }
    }

    public static class JavacBuilder extends JavaToolBuilder<JavacBuilder> {
        public ClassDir classDir;
        public SourcePath sourcePath;

        public JavacBuilder source(int version) {
            return opts("--source", Integer.toString(version));
        }

        public JavacBuilder class_dir(Path classDir) {
            this.classDir = ClassDir.of(classDir);
            return this;
        }

        public JavacBuilder class_dir(ClassDir classDir) {
            this.classDir = classDir;
            return this;
        }

        public JavacBuilder source_path(List<SourcePathEntry> sourcePaths) {
            this.sourcePath = SourcePath.ofOrUse(this.sourcePath).add(sourcePaths);
            return this;
        }

        public JavacBuilder source_path(SourcePathEntry... sourcePathEntries) {
            return source_path(List.of(sourcePathEntries));
        }

        public JavacBuilder source_path(SourcePath sourcePath) {
            return source_path(sourcePath.entries);
        }
    }

    public static JavacBuilder javac(JavacBuilder javacBuilder) {
        try {
            if (javacBuilder.classDir == null) {
                javacBuilder.classDir = ClassDir.temp();
            } else {
                javacBuilder.classDir.clean();
            }
            javacBuilder.opts("-d", javacBuilder.classDir.path().toString());

            if (javacBuilder.classPath != null) {
                javacBuilder.opts("--class-path", javacBuilder.classPath.charSeparated());
            }

            javacBuilder.opts("--source-path", javacBuilder.sourcePath.charSeparated());

            DiagnosticListener<JavaFileObject> diagnosticListener =
                    (diagnostic) -> {
                        if (!diagnostic.getKind().equals(Diagnostic.Kind.NOTE)) {
                            System.out.println(
                                    diagnostic.getKind()
                                            + " "
                                            + diagnostic.getLineNumber()
                                            + ":"
                                            + diagnostic.getColumnNumber()
                                            + " "
                                            + diagnostic.getMessage(null));
                        }
                    };

            JavaCompiler javac = javax.tools.ToolProvider.getSystemJavaCompiler();
            JavaCompiler.CompilationTask compilationTask =
                    (javac.getTask(
                            new PrintWriter(System.err),
                            javac.getStandardFileManager(diagnosticListener, null, null),
                            diagnosticListener,
                            javacBuilder.opts,
                            null,
                            javacBuilder.sourcePath.javaFiles().map(JavaSourceFile::new).toList()));
            ((com.sun.source.util.JavacTask) compilationTask).generate();
            return javacBuilder;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JavacBuilder javac(Consumer<JavacBuilder> javacBuilderConsumer) {
        JavacBuilder javacBuilder = new JavacBuilder();
        javacBuilderConsumer.accept(javacBuilder);
        return javac(javacBuilder);
    }

    public static class JavaBuilder extends JavaToolBuilder<JavaBuilder> {
        public String mainClass;
        public DirPath libraryPath;
        public boolean startOnFirstThread;
        public List<String> args = new ArrayList<>();

        public JavaBuilder enable_native_access(String module) {
            return opts("--enable-native-access=" + module);
        }

        public JavaBuilder args(List<String> args) {
            this.args.addAll(args);
            return self();
        }

        public JavaBuilder args(String... args) {
            args(Arrays.asList(args));
            return self();
        }

        public JavaBuilder main_class(String mainClass) {
            this.mainClass = mainClass;
            return this;
        }

        public JavaBuilder library_path(List<DirPathHolder<?>> libraryPathEntries) {
            this.libraryPath = DirPath.ofOrUse(this.libraryPath).add(libraryPathEntries);
            return this;
        }

        public JavaBuilder library_path(DirPath libraryPathEntries) {
            this.libraryPath = DirPath.ofOrUse(this.libraryPath).add(libraryPathEntries);
            return this;
        }

        public JavaBuilder library_path(DirPathHolder<?>... libraryPathEntries) {
            return this.library_path(List.of(libraryPathEntries));
        }

        public JavaBuilder start_on_first_thread() {
            this.startOnFirstThread =   true;
            return this;
        }
    }

    public static JavaBuilder java(JavaBuilder javaBuilder) {
        List<String> execOpts = new ArrayList<>();
        execOpts.add(javaBuilder.jdk.path().resolve("bin/java").toString());
        if (javaBuilder.startOnFirstThread){
            execOpts.add("-XstartOnFirstThread");
        }
        execOpts.addAll(javaBuilder.opts);
        if (javaBuilder.classPath != null) {
            execOpts.addAll(List.of("--class-path", javaBuilder.classPath.charSeparated()));
        }
        if (javaBuilder.libraryPath != null) {
            execOpts.add("-Djava.library.path=" + javaBuilder.libraryPath.charSeparated());
        }
        execOpts.add(javaBuilder.mainClass);
        execOpts.addAll(javaBuilder.args);

        try {
            var processBuilder = new ProcessBuilder().inheritIO().command(execOpts);
            var process = processBuilder.start();
            if (javaBuilder.verbose) {
                print(execOpts);
            }
            process.waitFor();
        } catch (InterruptedException | IOException ie) {
            System.out.println(ie);
        }

        return javaBuilder;
    }

    public static JavaBuilder java(Consumer<JavaBuilder> javaBuilderConsumer) {
        JavaBuilder javaBuilder = new JavaBuilder();
        javaBuilderConsumer.accept(javaBuilder);
        return java(javaBuilder);
    }

    public static JavaBuilder javaBuilder() {
        return new JavaBuilder();
    }

    public static class FormatBuilder extends Bldr.Builder<FormatBuilder> {
        public Bldr.SourcePath sourcePath;

        public FormatBuilder source_path(List<Bldr.SourcePathEntry> sourcePaths) {
            this.sourcePath = Bldr.SourcePath.ofOrUse(this.sourcePath).add(sourcePaths);
            return this;
        }

        public FormatBuilder source_path(Bldr.SourcePathEntry... sourcePaths) {
            return source_path(List.of(sourcePaths));
        }
    }

    public static void format(RepoDir repoDir, Consumer<FormatBuilder> formatBuilderConsumer) {
        var formatBuilder = new FormatBuilder();
        formatBuilderConsumer.accept(formatBuilder);
        var classPathEntries = repoDir.classPathEntries("com.google.googlejavaformat/google-java-format");

        java($ -> $
                .verbose()
                .enable_preview()
                .enable_native_access("ALL-UNNAMED")
                .add_exports("java.base", "jdk.internal", "ALL-UNNAMED")
                .add_exports(
                        "jdk.compiler",
                        List.of(
                                "com.sun.tools.javac.api",
                                "com.sun.tools.javac.code",
                                "com.sun.tools.javac.file",
                                "com.sun.tools.javac.main",
                                "com.sun.tools.javac.parser",
                                "com.sun.tools.javac.tree",
                                "com.sun.tools.javac.util"),
                        "ALL-UNNAMED")
                .class_path(classPathEntries)
                .main_class("com.google.googlejavaformat.java.Main")
                .args("-r")
                .args(formatBuilder.sourcePath.javaFiles().map(Path::toString).toList()));
    }

    public static class TestNGBuilder extends Bldr.Builder<TestNGBuilder> {
        public Bldr.SourcePath sourcePath;
        public Bldr.ClassPath classPath;
        private SuiteBuilder suiteBuilder;
        private JarFile testJar;

        public TestNGBuilder class_path(List<Bldr.ClassPathEntryProvider> classPathEntryProviders) {
            this.classPath = Bldr.ClassPath.ofOrUse(this.classPath).add(classPathEntryProviders);
            return this;
        }

        public TestNGBuilder class_path(Bldr.ClassPathEntryProvider... classPathEntryProviders) {
            class_path(List.of(classPathEntryProviders));
            return this;
        }

        public TestNGBuilder source_path(List<Bldr.SourcePathEntry> sourcePathEntries) {
            this.sourcePath = Bldr.SourcePath.ofOrUse(this.sourcePath).add(sourcePathEntries);
            return this;
        }

        public TestNGBuilder source_path(Bldr.SourcePathEntry... sourcePathEntries) {
            return source_path(List.of(sourcePathEntries));
        }

        public TestNGBuilder testJar(JarFile testJar) {
            this.testJar = testJar;
            return self();
        }

        public static class SuiteBuilder {
            String name;

            SuiteBuilder name(String name) {
                this.name = name;
                return this;
            }

            List<TestBuilder> testBuilders = new ArrayList<>();

            public static class TestBuilder {
                String name;
                List<String> classNames;

                TestBuilder name(String name) {
                    this.name = name;
                    return this;
                }

                public TestBuilder classes(List<String> classNames) {
                    this.classNames = this.classNames == null ? new ArrayList<>() : this.classNames;
                    this.classNames.addAll(classNames);
                    return this;
                }

                public TestBuilder classes(String... classNames) {
                    return classes(List.of(classNames));
                }
            }

            public void test(String testName, Consumer<TestBuilder> testBuilderConsumer) {
                TestBuilder testBuilder = new TestBuilder();
                testBuilder.name(testName);
                testBuilderConsumer.accept(testBuilder);
                testBuilders.add(testBuilder);
            }
        }

        public TestNGBuilder suite(String suiteName, Consumer<SuiteBuilder> suiteBuilderConsumer) {
            this.suiteBuilder = new SuiteBuilder();
            suiteBuilder.name(suiteName);
            suiteBuilderConsumer.accept(suiteBuilder);
            return self();
        }
    }

    public static void testng(RepoDir repoDir, Consumer<TestNGBuilder> testNGBuilderConsumer) {
        var testNGBuilder = new TestNGBuilder();
        testNGBuilderConsumer.accept(testNGBuilder);

        var text =
                XMLNode.create(
                                "suite",
                                $ -> {
                                    $.attr("name", testNGBuilder.suiteBuilder.name);
                                    testNGBuilder.suiteBuilder.testBuilders.forEach(
                                            tb -> {
                                                $.element(
                                                        "test",
                                                        $$ ->
                                                                $$.attr("name", tb.name)
                                                                        .element(
                                                                                "classes",
                                                                                $$$ ->
                                                                                        tb.classNames.forEach(
                                                                                                className ->
                                                                                                        $$$.element(
                                                                                                                "class",
                                                                                                                $$$$ -> $$$$.attr("name", className)))));
                                            });
                                })
                        .toString();

        TestNGSuiteFile testNGSuiteFile = Bldr.TestNGSuiteFile.containing(text);
        var mavenJars = repoDir.classPathEntries("org.testng/testng", "org.slf4j/slf4j-api");


        var testJar =
                testNGBuilder.testJar.create(
                        $ ->
                                $.javac(
                                        $$ ->
                                                $$.source(24)
                                                        .enable_preview()
                                                        .class_path(testNGBuilder.classPath, mavenJars)
                                                        .source_path(testNGBuilder.sourcePath)));

        java(
                $ ->
                        $.enable_preview()
                                .add_exports("java.base", "jdk.internal", "ALL-UNNAMED")
                                .enable_native_access("ALL-UNNAMED")
                                .class_path(testNGBuilder.classPath, mavenJars, testJar)
                                .main_class("org.testng.TestNG")
                                .args(testNGSuiteFile.path().toString()));
    }

    public static class JarBuilder extends Builder<JarBuilder> {
        public JarFile jar;
        public JavacBuilder javacBuilder;
        public DirPath dirList;

        public JarBuilder jar(JarFile jar) {
            this.jar = jar;
            return self();
        }

        public JarBuilder javac(JavacBuilder javacBuilder) {
            this.javacBuilder = Bldr.javac(javacBuilder);
            this.dirList =
                    (this.dirList == null)
                            ? DirPath.of().add(this.javacBuilder.classDir)
                            : this.dirList.add(this.javacBuilder.classDir);
            return self();
        }

        public JarBuilder javac(Consumer<JavacBuilder> javacBuilderConsumer) {
            this.javacBuilder = new JavacBuilder();
            javacBuilderConsumer.accept(this.javacBuilder);
            return javac(this.javacBuilder);
        }

        public <P extends DirPathHolder<P>> JarBuilder dir_list(P holder) {
            DirPath.ofOrUse(this.dirList).add(holder);
            return self();
        }
    }

    public static JarFile jar(JarBuilder jarBuilder) {
        try {
            List<RootDirAndSubPath> pathsToJar = new ArrayList<>();
            var jarStream = new JarOutputStream(Files.newOutputStream(jarBuilder.jar.path()));
            jarBuilder.dirList.entries.forEach(
                    root ->
                            root.findFiles()
                                    .map(path -> new RootDirAndSubPath(root, path))
                                    .forEach(pathsToJar::add));
            pathsToJar.stream()
                    .sorted(Comparator.comparing(RootDirAndSubPath::path))
                    .forEach(
                            rootAndPath -> {
                                try {
                                    var entry = new JarEntry(rootAndPath.relativize().toString());
                                    entry.setTime(Files.getLastModifiedTime(rootAndPath.path()).toMillis());
                                    jarStream.putNextEntry(entry);
                                    Files.newInputStream(rootAndPath.path()).transferTo(jarStream);
                                    jarStream.closeEntry();
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            });
            jarStream.finish();
            jarStream.close();
            return jarBuilder.jar;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JarFile jar(Consumer<JarBuilder> jarBuilderConsumer) {
        JarBuilder jarBuilder = new JarBuilder();
        jarBuilderConsumer.accept(jarBuilder);
        return jar(jarBuilderConsumer);
    }

    public static class CMakeBuilder extends Builder<CMakeBuilder> {
        public List<String> libraries = new ArrayList<>();
        public CMakeBuildDir cmakeBuildDir;
        public Dir sourceDir;
        private Path output;
        public BuildDir copyToDir;
        public List<String> opts = new ArrayList<>();

        public CMakeBuilder opts(List<String> opts) {
            this.opts.addAll(opts);
            return self();
        }

        public CMakeBuilder opts(String... opts) {
            opts(Arrays.asList(opts));
            return self();
        }

        public CMakeBuilder() {
            opts.add("cmake");
        }

        public CMakeBuilder build_dir(CMakeBuildDir cmakeBuildDir) {
            this.cmakeBuildDir = cmakeBuildDir;
            opts("-B", cmakeBuildDir.path.toString());
            return this;
        }

        public CMakeBuilder copy_to(BuildDir copyToDir) {
            this.copyToDir = copyToDir;
            opts("-DHAT_TARGET=" + this.copyToDir.path().toString());
            return this;
        }

        public CMakeBuilder source_dir(Dir sourceDir) {
            this.sourceDir = sourceDir;
            opts("-S", sourceDir.path().toString());
            return this;
        }

        public CMakeBuilder build(CMakeBuildDir cmakeBuildDir) {
            this.cmakeBuildDir = cmakeBuildDir;
            opts("--build", cmakeBuildDir.path().toString());
            return this;
        }
    }

    public static void cmake(Consumer<CMakeBuilder> cmakeBuilderConsumer) {
        CMakeBuilder cmakeBuilder = new CMakeBuilder();
        cmakeBuilderConsumer.accept(cmakeBuilder);
        cmakeBuilder.cmakeBuildDir.create();
        try {
            var processBuilder = new ProcessBuilder().inheritIO().command(cmakeBuilder.opts);
            var process = processBuilder.start();
            if (cmakeBuilder.verbose) {
                print(cmakeBuilder.opts);
            }
            process.waitFor();
        } catch (InterruptedException | IOException ie) {
            System.out.println(ie);
        }
    }

    static Path unzip(Path in, Path dir) {
        try {
            Files.createDirectories(dir);
            ZipFile zip = new ZipFile(in.toFile());
            zip.entries()
                    .asIterator()
                    .forEachRemaining(
                            entry -> {
                                try {
                                    String currentEntry = entry.getName();

                                    Path destFile = dir.resolve(currentEntry);
                                    // destFile = new File(newPath, destFile.getName());
                                    Path destinationParent = destFile.getParent();
                                    Files.createDirectories(destinationParent);
                                    // create the parent directory structure if needed

                                    if (!entry.isDirectory()) {
                                        zip.getInputStream(entry).transferTo(Files.newOutputStream(destFile));
                                    }
                                } catch (IOException ioe) {
                                    throw new RuntimeException(ioe);
                                }
                            });
            zip.close();

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return dir;
    }

    public static class JExtractBuilder extends Builder<JExtractBuilder> {
        public List<String> compileFlags = new ArrayList<>();
        public List<Path> libraries = new ArrayList<>();
        public List<Path> headers = new ArrayList<>();
        private String targetPackage;
        private BuildDir output;

        public JExtractBuilder target_package(String targetPackage) {
            this.targetPackage = targetPackage;
            return this;
        }

        public JExtractBuilder output(BuildDir output) {
            this.output = output;
            return this;
        }

        public JExtractBuilder library(Path... libraries) {
            this.libraries.addAll(Arrays.asList(libraries));
            return this;
        }

        public JExtractBuilder compile_flag(String... compileFlags) {
            this.compileFlags.addAll(Arrays.asList(compileFlags));
            return this;
        }

        public JExtractBuilder header(Path header) {
            this.headers.add(header);
            return this;
        }
    }

    public static void jextract(Jextract executable, Consumer<JExtractBuilder> jextractBuilderConsumer) {
        var exePath = executable.path;
        var homePath = exePath.getParent().getParent();

        JExtractBuilder jExtractBuilder = new JExtractBuilder();
        jextractBuilderConsumer.accept(jExtractBuilder);
        List<String> opts = new ArrayList<>();
        opts.add(executable.path().toString());

        if (jExtractBuilder.targetPackage != null) {
            opts.addAll(List.of("--target-package", jExtractBuilder.targetPackage));
        }
        if (jExtractBuilder.output != null) {
            jExtractBuilder.output.create();
            opts.addAll(List.of("--output", jExtractBuilder.output.path().toString()));
        }
        for (Path library : jExtractBuilder.libraries) {
            opts.addAll(List.of("--library", ":" + library));
        }

        for (Path header : jExtractBuilder.headers) {
            opts.add(header.toString());
        }

        if (jExtractBuilder.compileFlags != null && !jExtractBuilder.compileFlags.isEmpty()) {
            jExtractBuilder.output.textFile("compile_flags.txt", jExtractBuilder.compileFlags);
        }

        if (jExtractBuilder.verbose) {
            StringBuilder sb = new StringBuilder();
            opts.forEach(opt -> (sb.isEmpty() ? sb : sb.append(" ")).append(opt));
            println(sb);
        }
        var processBuilder = new ProcessBuilder();
        if (jExtractBuilder.output != null) {
            processBuilder.directory(jExtractBuilder.output.path().toFile());
        }
        processBuilder.inheritIO().command(opts);
        try {
            processBuilder.start().waitFor();
        } catch (InterruptedException | IOException ie) {
            throw new RuntimeException(ie);
        }
    }

    public record SearchableTextFile(Path path) implements TextFile {
        static SearchableTextFile of(Path path) {
            return new SearchableTextFile(path);
        }
        public Stream<Line> lines() {
            try {
                int num[] = new int[]{1};
                return Files.readAllLines(path(), StandardCharsets.UTF_8).stream()
                        .map(line -> new Line(line, num[0]++));
            } catch (IOException ioe) {
                System.out.println(ioe);
                return new ArrayList<Line>().stream();
            }
        }

        public boolean grep(Pattern pattern) {
            return lines().anyMatch(line -> pattern.matcher(line.line).matches());
        }

        public boolean hasSuffix(String... suffixes) {
            var suffixSet = Set.of(suffixes);
            int dotIndex = path().toString().lastIndexOf('.');
            return dotIndex == -1 || suffixSet.contains(path().toString().substring(dotIndex + 1));
        }
    }

    public record Line(String line, int num) {
        public boolean grep(Pattern pattern) {
            return pattern.matcher(line()).matches();
        }
    }

    public static Path curl(URL url, Path file) {
        try {
            println("Downloading " + url + "->" + file);
            url.openStream().transferTo(Files.newOutputStream(file));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return file;
    }

    public static Optional<Path> which(String execName) {
        // which and whereis had issues.
        return Arrays.asList(System.getenv("PATH").split(File.pathSeparator)).stream()
                .map(dirName -> Path.of(dirName).resolve(execName).normalize())
                .filter(Files::isExecutable)
                .findFirst();
    }

    public static boolean canExecute(String execName) {
        return which(execName).isPresent();
    }

    public static Path untar(Path tarFile, Path dir) {
        try {
            new ProcessBuilder()
                    .inheritIO()
                    .command("tar", "xvf", tarFile.toString(), "--directory", tarFile.getParent().toString())
                    .start()
                    .waitFor();
            return dir;
        } catch (
                InterruptedException
                        e) { // We get IOException if the executable not found, at least on Mac so interuppted
            // means it exists
            return null;
        } catch (IOException e) { // We get IOException if the executable not found, at least on Mac
            // throw new RuntimeException(e);
            return null;
        }
    }

    public static Path requireJExtract(Dir thirdParty) {
        var optional = fromPATH("jextract");
        if (optional.isPresent()) {
            println("Found jextract in PATH");
            return optional.get().getParent().getParent(); // we want the 'HOME' dir
        }
        println("No jextract in PATH");
        URL downloadURL = null;
        var extractVersionMaj = "22";
        var extractVersionMin = "5";
        var extractVersionPoint = "33";

        var nameArchTuple =
                switch (os.name()) {
                    case OS.MacName -> "macos";
                    default -> os.name().toLowerCase();
                }
                        + '-'
                        + os.arch();

        try {
            downloadURL =
                    new URI(
                            "https://download.java.net/java/early_access"
                                    + "/jextract/"
                                    + extractVersionMaj
                                    + "/"
                                    + extractVersionMin
                                    + "/openjdk-"
                                    + extractVersionMaj
                                    + "-jextract+"
                                    + extractVersionMin
                                    + "-"
                                    + extractVersionPoint
                                    + "_"
                                    + nameArchTuple
                                    + "_bin.tar.gz")
                            .toURL();
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
        URL finalDownloadURL = downloadURL;

        println("... attempting download from" + downloadURL);
        var jextractTar = thirdParty.path("jextract.tar");

        if (!isRegularFile(jextractTar)) { // Have we downloaded already?
            jextractTar = curl(finalDownloadURL, jextractTar); // if not
        }

        var jextractHome = thirdParty.path("jextract-22");
        if (!isDirectory(jextractHome)) {
            untar(jextractTar, jextractHome);
        }
        return jextractHome;
    }

    public static Optional<Path> fromPATH(String name) {
        return Arrays.stream(System.getenv("PATH").split(File.pathSeparator))
                .map(dirName -> Path.of(dirName).resolve(name).normalize())
                .filter(Files::isExecutable).findFirst();
    }


    public static <T extends PathHolder> T assertExists(T testme) {
        if (Files.exists(testme.path())) {
            return testme;
        } else {
            throw new IllegalStateException("FAILED: " + testme.path() + " does not exist");
        }
    }

    public static <T extends Path> T assertExists(T path) {
        if (Files.exists(path)) {
            return path;
        } else {
            throw new IllegalStateException("FAILED: " + path + " does not exist");
        }
    }

    void main(String[] args) {
        var bldrDir = Dir.current().parent().parent().parent();
        var buildDir = BuildDir.of(bldrDir.path("build")).create();

        jar(
                $ ->
                        $.jar(buildDir.jarFile("bldr.jar"))
                                .javac(
                                        $$ ->
                                                $$.source(24)
                                                        .enable_preview()
                                                        .add_exports(
                                                                "java.base",
                                                                List.of("jdk.internal", "jdk.internal.vm.annotation"),
                                                                "ALL-UNNAMED")
                                                        .class_dir(buildDir.classDir("bld.jar.classes"))
                                                        .source_path(bldrDir.sourceDir("src/main/java"))));
    }
}

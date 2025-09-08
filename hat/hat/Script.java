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
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
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

import static java.lang.IO.print;
import static java.lang.IO.println;

public class Script {
    public sealed interface PathHolder permits ClassPathEntry, DirPathHolder, FilePathHolder, SourcePathEntry {
        default Path path(String subdir) {
            return path().resolve(subdir);
        }

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
        default boolean failsToMatch(String pattern) {
            return !pathMatcher(Pattern.compile(pattern)).matches();
        }

        boolean exists();

        Path path();
    }

    public sealed interface DirPathHolder<T extends DirPathHolder<T>> extends PathHolder
            permits BuildDirHolder, DirEntry, SourceDir {

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
            return Files.isDirectory(path());
        }

        default BuildDir buildDir(String name) {
            return BuildDir.of(path().resolve(name));
        }

        default SourceDir sourceDir(String s) {
            return SourceDir.of(path().resolve(s));
        }

        default XMLFile xmlFile(String s) {
            return XMLFile.of(path().resolve(s));
        }

       }

    public sealed interface FilePathHolder extends PathHolder  {
        default boolean exists() {
            return Files.isRegularFile(path());
        }
    }

    public non-sealed interface Executable extends FilePathHolder {
        default boolean exists() {
            return Files.exists(path()) && Files.isRegularFile(path()) && Files.isExecutable(path());
        }
    }


    public interface ClassPathEntryProvider {
        List<ClassPathEntry> classPathEntries();
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

    public record SourcePath(List<SourceDir> entries)
            implements PathHolderList<SourceDir> {
        public static SourcePath of() {
            return new SourcePath(new ArrayList<>());
        }

        public static SourcePath ofOrUse(SourcePath sourcePath) {
            return sourcePath == null ? of() : sourcePath;
        }

        public SourcePath add(List<SourceDir> sourcePathEntries) {
            entries.addAll(sourcePathEntries);
            return this;
        }

        public SourcePath add(SourceDir... sourcePathEntries) {
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

    public sealed interface BuildDirHolder<T extends BuildDirHolder<T>> extends DirPathHolder<T> {
        T create();

        T remove();

        default T clean() {
            remove();
            return create();
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


    public record DirEntry(Path path) implements DirPathHolder<DirEntry> {
        public static DirEntry of(Path path) {
            return new DirEntry(path);
        }

        public static DirEntry of(String string) {
            return of(Path.of(string));
        }

        public static DirEntry ofExisting(String string) {
            return of(assertExists(Path.of(string)));
        }

        public static DirEntry current() {
            return of(Path.of(System.getProperty("user.dir")));
        }

        public DirEntry parent() {
            return of(path().getParent());
        }

        public DirEntry dir(String subdir) {
            return DirEntry.of(path(subdir));
        }
        public DirEntry optionalDir(String subdir) {
            var possible =  DirEntry.of(path(subdir));
            return possible.exists()? possible: null;
        }

        public FileEntry file(String fileName) {
            return FileEntry.of(path(fileName));
        }

        public DirEntry existingDir(String subdir) {
            return assertExists(DirEntry.of(path(subdir)));
        }

        public Stream<DirEntry> subDirs() {
            return Stream.of(Objects.requireNonNull(path().toFile().listFiles(File::isDirectory)))
                    .map(d -> DirEntry.of(d.getPath()));
        }

        public BuildDir existingBuildDir(String subdir) {
            return assertExists(BuildDir.of(path(subdir)));
        }
        public MavenStyleProject mavenStyleProject(BuildDir buildDir,JarFile jarFile,ClassPathEntryProvider ... classPathEntryProviders ) {
            return Script.mavenStyleProject(buildDir,this, jarFile,classPathEntryProviders);
        }
    }

    public interface SourcePathEntryProvider {
        List<SourcePathEntry> sourcePathEntries();
    }

    public sealed interface SourcePathEntry extends PathHolder, SourcePathEntryProvider {
    }

    public record SourceDir(Path path) implements SourcePathEntry, DirPathHolder<SourceDir> {
        public static SourceDir of(Path path) {
            return new SourceDir(path);
        }
        public static SourceDir of(DirEntry dirEntry) {
            return new SourceDir(dirEntry.path());
        }
        public static SourceDir of(BuildDir buildDir) {
            return new SourceDir(buildDir.path());
        }

        public Stream<Path> javaFiles() {
            return findFilesBySuffix(".java");
        }

        @Override
        public List<SourcePathEntry> sourcePathEntries() {
            return List.of(this);
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

        public ClassPathEntryProvider jarFiles(String... names) {
            var classPath = ClassPath.of();
            Stream.of(names).forEach(name -> classPath.add(JarFile.of(path().resolve(name))));
            return classPath;
        }

        public ClassDir classDir(String name) {
            return ClassDir.of(path().resolve(name));
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

        public SearchableTextFile textFile(String file, Consumer<StringBuilder> stringBuilderConsumer) {
            SearchableTextFile textFile = SearchableTextFile.of(path().resolve(file));
            var sb = new StringBuilder();
            stringBuilderConsumer.accept(sb);
            try {
                Files.writeString(textFile.path, sb.toString());
                return textFile;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }


        public CMakeLists cmakeLists(Consumer<StringBuilder> stringBuilderConsumer) {
            var sb = new StringBuilder();
            stringBuilderConsumer.accept(sb);
            var ret = CMakeLists.of(path().resolve("CMakeLists.txt"));
            try {
                Files.writeString(ret.path, sb.toString());
                return ret;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public MavenStyleProject mavenStyleBuild(DirEntry dirEntry, String jarName, ClassPathEntryProvider ...classPathEntryProvider) {
            return mavenStyleProject(this, dirEntry, jarFile(jarName), classPathEntryProvider).build();
        }
    }

    public record FileEntry(Path path) implements FilePathHolder {
        public static FileEntry of(Path path) {
            return new FileEntry(path);
        }
    }

    public record JarFile(Path path) implements ClassPathEntry, FilePathHolder {
        public static JarFile of(Path path) {
            return new JarFile(path);
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(this);
        }

        public boolean contains(String entryName) {
            java.util.jar.JarFile jarFile = null;
            try {
                jarFile = new java.util.jar.JarFile(path.toFile());
                JarEntry entry = jarFile.getJarEntry(entryName);
                return entry != null;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        public List<JarEntry> select(Regex regex) {
            java.util.jar.JarFile jarFile = null;
            try {
                jarFile = new java.util.jar.JarFile(path.toFile());
                return jarFile.stream().filter(je->regex.matches(je.getName())).toList();

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
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

    public static final class CMakeLists implements SourceFile {
        Path path;

        CMakeLists(Path path) {
            this.path = path;
        }

        public static CMakeLists of(Path path) {
            return new CMakeLists(path);
        }

        @Override
        public Path path() {
            return path;
        }
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
            super(path.toUri(), Kind.SOURCE);
            this.path = path;
        }

        @Override
        public Path path() {
            return path;
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



    public record XMLFile(Path path) implements TextFile {
        public static XMLFile of(Path path) {
            return new XMLFile(path);
        }

        public static XMLFile containing(String text) {
            return XMLFile.of(TextFile.tempContaining("xml", text));
        }
    }



    public interface OS {
        String arch();

        String name();

        String version();

        record Linux(String arch, String name, String version) implements OS {
        }

        record Mac(String arch, String name, String version) implements OS {
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

    public record Java(String version, DirEntry home, int specVersion) {
    }

    public static Java java =
            new Java(
                    System.getProperty("java.version"),
                    DirEntry.of(System.getProperty("java.home")),
                    Integer.parseInt(System.getProperty("java.specification.version"))
            );

    public record User(DirEntry home, DirEntry pwd) {
    }

    public static User user =
            new User(DirEntry.of(System.getProperty("user.home")), DirEntry.of(System.getProperty("user.dir")));


    public abstract sealed static class Builder<T extends Builder<T>> permits CMakeBuilder, JarBuilder, JarBuilder.ManifestBuilder, JavaOpts {
        public Builder<?> parent;
        public boolean verbose;
        public boolean quiet;

        @SuppressWarnings("unchecked")
        T self() {
            return (T) this;
        }

        protected T dontCallThisCopy(T other) {
            this.verbose = other.verbose;
            this.quiet = other.quiet;
            return self();
        }

        public T quiet(boolean quiet) {
            this.quiet = quiet;
            return self();
        }

        public T quiet() {
            quiet(true);
            return self();
        }

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

        Builder(Builder<?> parent) {
            this.parent = parent;
        }

        Builder() {
            this(null);
        }

        public T mac(Consumer<OS.Mac> macConsumer) {
            if (Script.os instanceof OS.Mac mac) {
                macConsumer.accept(mac);
            }
            return self();
        }

        public T linux(Consumer<OS.Linux> linuxConsumer) {
            if (Script.os instanceof OS.Linux linux) {
                linuxConsumer.accept(linux);
            }
            return self();
        }

        public T os(Consumer<OS.Mac> macConsumer, Consumer<OS.Linux> linuxConsumer) {
            switch (Script.os) {
                case OS.Linux linux -> linuxConsumer.accept(linux);
                case OS.Mac mac -> macConsumer.accept(mac);
                default -> throw new IllegalStateException("Unexpected value: " + Script.os);
            }
            return self();
        }
    }

    public abstract static sealed class Result<T extends Builder<T>> implements DiagnosticListener <JavaFileObject>  permits  JarResult, JavaResult, JavacResult {
        public boolean ok;
        public T builder;
        public List<Diagnostic<? extends JavaFileObject>> diagnostics = new ArrayList<>();
        Result(T builder) {
            this.builder = builder;
        }
        @Override public void report(Diagnostic<? extends JavaFileObject> diagnostic){
            if (diagnostic.getKind().equals(Diagnostic.Kind.ERROR)) {
                ok = false;
            }
           diagnostics.add(diagnostic);
        }
        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            diagnostics.forEach(diagnostic -> {
                        builder
                                .append("\n")
                                .append(switch (diagnostic.getKind()){
                                    case ERROR -> "\u001B[31mERR :";  //RED
                                    case WARNING -> "\u001B[33mWARN:"; //YELLOW
                                    case NOTE -> "\u001B[32mNOTE:"; // GREEN
                                    default -> "diagnostic.getKind()";
                                })
                                .append(" ");
                        if (diagnostic.getSource() instanceof JavaSourceFile sourceFile) {
                            builder.
                                    append(sourceFile.path())
                                    .append(" ");
                        }
                        builder
                                .append(diagnostic.getLineNumber())
                                .append(":")
                                .append(diagnostic.getColumnNumber())
                                .append("\n")
                                .append(diagnostic.getMessage(null))
                                .append("\u001B[0m");

                    }
            );
            return builder.toString();
        }

        public void note(JavaSourceFile javaSourceFile, String message) {
            diagnostics.add(new Diagnostic<JavaFileObject>() {
                                       @Override
                                       public Kind getKind() {
                                           return Kind.NOTE;
                                       }

                                       @Override
                                       public JavaFileObject getSource() {
                                           return javaSourceFile;
                                       }

                                       @Override
                                       public long getPosition() {
                                           return 0;
                                       }

                                       @Override
                                       public long getStartPosition() {
                                           return 0;
                                       }

                                       @Override
                                       public long getEndPosition() {
                                           return 0;
                                       }

                                       @Override
                                       public long getLineNumber() {
                                           return 0;
                                       }

                                       @Override
                                       public long getColumnNumber() {
                                           return 0;
                                       }

                                       @Override
                                       public String getCode() {
                                           return "";
                                       }

                                       @Override
                                       public String getMessage(Locale locale) {
                                           return message;
                                       }
                                   }
            );
            //    println("Excluded " + javaSourceFile);
        }

    }

    public static class StringList {
        public List<String> strings = new ArrayList<>();

        StringList() {
        }

        StringList(StringList stringList) {
            add(stringList);
        }

        StringList(List<String> strings) {
            add(strings);
        }

        StringList(String... strings) {
            add(strings);
        }

        public StringList add(List<String> strings) {
            this.strings.addAll(strings);
            return this;
        }

        public StringList add(String... strings) {
            add(Arrays.asList(strings));
            return this;
        }

        public StringList add(StringList stringList) {
            add(stringList.strings);
            return this;
        }

        public String spaceSeparated() {
            StringBuilder stringBuilder = new StringBuilder();
            strings.forEach(opt -> stringBuilder.append(stringBuilder.isEmpty() ? "" : " ").append(opt));
            return stringBuilder.toString();
        }
    }


    public static sealed class JavaOpts<T extends JavaOpts<T>> extends Builder<T> {
        public DirEntry jdk = java.home;
        public Boolean enablePreview;
        public StringList modules;
        protected boolean justShowCommandline;

        record FromModulePackageToModule(String fromModule, String pkg, String toModule) {
        }

        List<FromModulePackageToModule> exports;

        protected T dontCallThisCopy(T other) {
            super.dontCallThisCopy(other);
            if (other.jdk != null) {
                this.jdk = other.jdk;
            }
            if (other.enablePreview != null) {
                this.enablePreview = other.enablePreview;
            }
            if (other.modules != null) {
                this.modules = new StringList(other.modules);
            }
            if (other.exports != null) {
                this.exports = new ArrayList<>(other.exports);
            }

            return self();
        }

        public JavaOpts(Builder<?> parent) {
            super(parent);
        }

        public JavaOpts() {
            super();
        }

        static public JavaOpts<?> of() {
            return new JavaOpts<>();
        }

        public T jdk(DirEntry jdk) {
            this.jdk = jdk;
            return self();
        }

        public T add_exports(String fromModule, String pkg, String toModule) {
            if (this.exports == null) {
                this.exports = new ArrayList<>();
            }
            exports.add(new FromModulePackageToModule(fromModule, pkg, toModule));
            return self();
        }

        public T add_modules(String... modules) {
            if (this.modules == null) {
                this.modules = new StringList();
            }
            this.modules.add(modules);

            return self();
        }

        public T add_exports(String fromModule, List<String> packages, String toModule) {

            packages.forEach(p -> add_exports(fromModule, p, toModule));
            return self();
        }

        public T add_exports(String fromModule, String[] packages, String toModule) {
            return add_exports(fromModule, Arrays.asList(packages), toModule);
        }

        public T add_exports_to_all_unnamed(String fromModule, String... packages) {
            return add_exports(fromModule, Arrays.asList(packages), "ALL-UNNAMED");
        }

        public T enable_preview() {
            this.enablePreview = true;
            return self();
        }

        public T justShowCommandline(boolean justShowCommandline) {
            this.justShowCommandline=justShowCommandline;
            return self();
        }
        public T justShowCommandline() {
            return justShowCommandline(true);
        }

    }

    public abstract sealed static class JavaToolBuilder<T extends JavaToolBuilder<T>> extends JavaOpts<T> permits JavacBuilder, JavaBuilder {
        public ClassPath classPath;


        protected T dontCallThisCopy(T other) {
            super.dontCallThisCopy(other);
            if (other.classPath != null) {
                this.classPath = ClassPath.of().add(other.classPath);
            }
            return self();
        }

        public JavaToolBuilder(Builder<?> parent) {
            super(parent);
        }

        public JavaToolBuilder() {
            super();
        }

        public T class_path(List<ClassPathEntryProvider> classPathEntryProviders) {
            this.classPath = ClassPath.ofOrUse(this.classPath).add(classPathEntryProviders);
            return self();
        }

        public T class_path(ClassPathEntryProvider... classPathEntryProviders) {
            return class_path(List.of(classPathEntryProviders));
        }
    }

    public static final class JavacBuilder extends JavaToolBuilder<JavacBuilder> {
        public DirEntry mavenStyleRoot;
        public ClassDir classDir;
        public SourcePath sourcePath;
        public ClassPath modulePath;
        public SourcePath moduleSourcePath;
        public Integer source;
        public List<Predicate<JavaSourceFile>> exclusionFilters;

        protected JavacBuilder dontCallThisCopy(JavacBuilder other) {
            super.dontCallThisCopy(other);
            if (other.mavenStyleRoot != null) {
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to maven style dir");
            }
            if (other.sourcePath != null) {
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to a SourcePath");
            }
            if (other.moduleSourcePath != null) {
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to a ModuleSourcePath");
            }

            if (other.source != null) {
                this.source = other.source;
            }

            if (other.classPath != null) {
                ClassPath.ofOrUse(this.classPath).add(other.classPath);
            }
            return this;
        }

        public JavacBuilder source(int version) {
            this.source = version;
            return self();
        }

        public JavacBuilder current_source() {
            return source(Script.java.specVersion);
        }

        public JavacBuilder maven_style_root(DirEntry mavenStyleRoot) {
            this.mavenStyleRoot = mavenStyleRoot;
            return this;
        }

        public JavacBuilder class_dir(Path classDir) {
            this.classDir = ClassDir.of(classDir);
            return this;
        }

        public JavacBuilder class_dir(ClassDir classDir) {
            this.classDir = classDir;
            return this;
        }

        public JavacBuilder d(ClassDir classDir) {
            this.classDir = classDir;
            return this;
        }

        public JavacBuilder source_path(List<SourceDir> sourcePaths) {
            this.sourcePath = SourcePath.ofOrUse(this.sourcePath).add(sourcePaths);
            return this;
        }

        public JavacBuilder source_path(SourceDir... sourcePathEntries) {
            return source_path(List.of(sourcePathEntries));
        }

        public JavacBuilder source_path(SourcePath sourcePath) {
            return source_path(sourcePath.entries);
        }

        public JavacBuilder module_source_path(List<SourceDir> moduleSourcePathEntries) {
            this.moduleSourcePath = SourcePath.ofOrUse(this.moduleSourcePath).add(moduleSourcePathEntries);
            return this;
        }

        public JavacBuilder module_source_path(SourceDir... moduleSourcePathEntries) {
            return module_source_path(List.of(moduleSourcePathEntries));
        }

        public JavacBuilder module_source_path(SourcePath moduleSourcePath) {
            return module_source_path(moduleSourcePath.entries());
        }

        public JavacBuilder() {
            super();
        }

        public JavacBuilder(JarBuilder jarBuilder) {
            super(jarBuilder);
        }

        public JavacBuilder exclude(Predicate<JavaSourceFile> javaSourceFileFilter) {
            this.exclusionFilters = (this.exclusionFilters == null ? new ArrayList<>() : this.exclusionFilters);
            this.exclusionFilters.add(javaSourceFileFilter);
            return self();
        }
    }

    public static final class JavacResult extends Result<JavacBuilder> {
        StringList opts = new StringList();

        List<JavaSourceFile> sourceFiles = new ArrayList<>();
        List<JavaFileObject> classes = new ArrayList<>();
        public ClassDir classDir;

        JavacResult(JavacBuilder builder) {
            super(builder);
        }

    }

    public static JavacResult javac(JavacBuilder javacBuilder) {
        JavacResult result = new JavacResult(javacBuilder);

        try {
            if (javacBuilder.source != null) {
                result.opts.add("--source", javacBuilder.source.toString());
            }

            if (javacBuilder.enablePreview != null && javacBuilder.enablePreview) {
                result.opts.add("--enable-preview");
            }
            if (javacBuilder.modules != null) {
                javacBuilder.modules.strings.forEach(module ->
                        result.opts.add("--add-modules", module)
                );
            }

            if (javacBuilder.exports != null) {
                javacBuilder.exports.forEach(fpt -> {
                    result.opts.add("--add-exports=" + fpt.fromModule + "/" + fpt.pkg + "=" + fpt.toModule);
                });
            }

            result.classDir = javacBuilder.classDir == null ? ClassDir.temp() : javacBuilder.classDir;
            result.opts.add("-d", result.classDir.path().toString());
            if (javacBuilder.classPath != null) {
                result.opts.add("--class-path", javacBuilder.classPath.charSeparated());
            } else if (javacBuilder.modulePath != null) {
                //https://dev.java/learn/modules/building/
                result.opts.add("--module-path", javacBuilder.modulePath.charSeparated());
            }
            var mavenStyleRoot =
                    ((javacBuilder.parent instanceof JarBuilder jarBuilder) && jarBuilder.mavenStyleRoot instanceof DirEntry fromJarBuilder)
                            ? fromJarBuilder
                            : javacBuilder.mavenStyleRoot;


            if (mavenStyleRoot == null) {
                if (javacBuilder.sourcePath != null && !javacBuilder.sourcePath.entries.isEmpty()) {
                    result.opts.add("--source-path", javacBuilder.sourcePath.charSeparated());
                    result.sourceFiles.addAll(javacBuilder.sourcePath.javaFiles().map(JavaSourceFile::new).toList());
                } else if (javacBuilder.moduleSourcePath != null && !javacBuilder.moduleSourcePath.entries.isEmpty()) {
                    result.opts.add("--module-source-path", javacBuilder.moduleSourcePath.charSeparated());
                    result.sourceFiles.addAll(javacBuilder.moduleSourcePath.javaFiles().map(JavaSourceFile::new).toList());
                } else {
                    throw new RuntimeException("No source path or module source path specified");
                }
            } else {
                var sourcePath = SourcePath.of().add(SourceDir.of(mavenStyleRoot.path.resolve("src/main/java")));
                result.sourceFiles.addAll(sourcePath.javaFiles().map(JavaSourceFile::new).toList());
                if (result.sourceFiles.isEmpty()) {
                    throw new RuntimeException("No sources");
                }
                result.opts.add("--source-path", sourcePath.charSeparated());

                if (javacBuilder.sourcePath != null && !javacBuilder.sourcePath.entries.isEmpty()) {
                    throw new RuntimeException("You have specified --source-path AND provided maven_style_root ");
                }
            }
           result.ok = true;


            JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
            if (javacBuilder.exclusionFilters != null) {
                javacBuilder.exclusionFilters.forEach(p -> {
                    result.sourceFiles = result.sourceFiles.stream().filter(
                            javaSourceFile -> {
                                var kill = p.test(javaSourceFile);
                                if (kill) {
                                    result.note(javaSourceFile, "Excluded");
                                }
                                return !kill;
                            }
                    ).toList();
                });
            }
            if (javacBuilder.verbose || javacBuilder.parent instanceof JarBuilder jarBuilder && jarBuilder.verbose) {

                print("javac " + result.opts.spaceSeparated());
                result.sourceFiles.forEach(s -> print(s + " "));
                println("");
            }
            JavaCompiler.CompilationTask compilationTask =
                    (javac.getTask(
                            new PrintWriter(System.err),
                            javac.getStandardFileManager(result, null, null),
                            result,
                            result.opts.strings,
                            null,
                            result.sourceFiles
                    ));
            JavacTask javacTask = (JavacTask) compilationTask;

            javacTask.generate().forEach(javaFileObject -> {
                result.classes.add(javaFileObject);
            });
            if (!result.ok) {
                throw new RuntimeException("javac failed to compile "+result);
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JavacBuilder javacBuilder(Consumer<JavacBuilder> javacBuilderConsumer) {
        JavacBuilder javacBuilder = new JavacBuilder();
        javacBuilderConsumer.accept(javacBuilder);
        return javacBuilder;
    }

    public static JavacResult javac(Consumer<JavacBuilder> javacBuilderConsumer) {
        return javac(javacBuilder(javacBuilderConsumer));
    }

    public static final class JavaBuilder extends JavaToolBuilder<JavaBuilder> {
        public String mainClass;
        public DirPath libraryPath;
        public boolean startOnFirstThread;
        public StringList vmargs = new StringList();
        public StringList args = new StringList();
        public StringList nativeAccessModules = new StringList();
        private boolean headless;
        public boolean noModuleOp;


        public JavaBuilder enable_native_access(String module) {
            nativeAccessModules.add(module);
            return self();
        }

        public JavaBuilder enable_native_access_to_all_unnamed() {
            return enable_native_access("ALL-UNNAMED");
        }
        public JavaBuilder vmargs(List<String> args) {
            this.vmargs.add(args);
            return self();
        }

        public JavaBuilder vmargs(String... args) {
            vmargs(Arrays.asList(args));
            return self();
        }

        public JavaBuilder args(List<String> args) {
            this.args.add(args);
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
            this.startOnFirstThread = true;
            return this;
        }

        public void headless() {
            this.headless = true;
        }

        public void noModuleOp() {
            this.noModuleOp = true;
        }
    }

    public static final class JavaResult extends Result<JavaBuilder> {
        StringList opts = new StringList();

        JavaResult(JavaBuilder javaBuilder) {
            super(javaBuilder);
        }
    }

    public static JavaBuilder java(JavaBuilder javaBuilder) {
        JavaResult result = new JavaResult(javaBuilder);
        result.ok = true;
        result.opts.add(javaBuilder.jdk.path().resolve("bin/java").toString());
        if (javaBuilder.enablePreview != null && javaBuilder.enablePreview) {
            result.opts.add("--enable-preview");
        }
        if (javaBuilder.modules != null) {
            javaBuilder.modules.strings.forEach(module ->
                    result.opts.add("--add-modules", module)
            );
        }

        if (javaBuilder.exports != null) {
            javaBuilder.exports.forEach(fpt -> {
                result.opts.add("--add-exports=" + fpt.fromModule + "/" + fpt.pkg + "=" + fpt.toModule);
            });
        }
        if (javaBuilder.headless) {
            result.opts.add("-Dheadless=true");
        }
        if (javaBuilder.noModuleOp) {
            result.opts.add("-DnoModuleOp=true");
        }
        if (javaBuilder.startOnFirstThread) {
            result.opts.add("-XstartOnFirstThread");
        }

        javaBuilder.nativeAccessModules.strings.forEach(module ->
                result.opts.add("--enable-native-access=" + module)
        );

        if (javaBuilder.classPath != null) {
            result.opts.add("--class-path", javaBuilder.classPath.charSeparated());
        }
        if (javaBuilder.libraryPath != null) {
            result.opts.add("-Djava.library.path=" + javaBuilder.libraryPath.charSeparated());
        }
        result.opts.add(javaBuilder.vmargs.strings);
        result.opts.add(javaBuilder.mainClass);
        result.opts.add(javaBuilder.args.strings);

        if (javaBuilder.justShowCommandline) {
            println(result.opts.spaceSeparated());
            result.ok = false;
        }else {
            try {
                var processBuilder = new ProcessBuilder().inheritIO().command(result.opts.strings);
                var process = processBuilder.start();
                if (javaBuilder.verbose) {
                    println(result.opts.spaceSeparated());
                }
                process.waitFor();
                result.ok = (process.exitValue() == 0);
                if (!result.ok) {
                    println("java returned error " + process.exitValue());
                }

            } catch (InterruptedException | IOException ie) {
                result.ok = false;
                System.out.println(ie);
            }
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

    public static final class JarBuilder extends Builder<JarBuilder> {
        public static class Manifest {
            public String mainClass;
            public String[] classPath;
            public String version;
            public String createdBy;
            public String buildBy;

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

        public static final class ManifestBuilder extends Builder<ManifestBuilder> {

            Manifest manifest;

            public ManifestBuilder main_class(String mainClass) {
                this.manifest.mainClass = mainClass;
                return self();
            }

            public ManifestBuilder version(String version) {
                this.manifest.version = version;
                return self();
            }

            public ManifestBuilder created_by(String createdBy) {
                this.manifest.createdBy = createdBy;
                return self();
            }

            public ManifestBuilder build_by(String buildBy) {
                this.manifest.buildBy = buildBy;
                return self();
            }

            public ManifestBuilder class_path(String... classPath) {
                this.manifest.classPath = classPath;
                return self();
            }

            public ManifestBuilder class_path(ClassPathEntry... classPathEntries) {
                this.manifest.classPath = Stream.of(classPathEntries).map(classPathEntry -> classPathEntry.path().getFileName().toString()).toArray(String[]::new);
                return self();
            }

            ManifestBuilder(Manifest manifest) {
                this.manifest = manifest;
            }
        }

        public DirEntry mavenStyleRoot;
        public JarFile jar;
        public JavacResult javacResult;
        public DirPath dirList;
        public Manifest manifest;

        public JarBuilder jarFile(JarFile jar) {
            this.jar = jar;
            return self();
        }

        public JarBuilder maven_style_root(DirEntry mavenStyleRoot) {
            this.mavenStyleRoot = mavenStyleRoot;
            return this;
        }

        public JarBuilder manifest(Consumer<ManifestBuilder> manifestBuilderConsumer) {
            this.manifest = this.manifest == null ? new Manifest() : this.manifest;
            var manifestBuilder = new ManifestBuilder(manifest);
            manifestBuilderConsumer.accept(manifestBuilder);
            return self();
        }

        private JarBuilder javac(JavacBuilder javacBuilder) {
            this.javacResult = Script.javac(javacBuilder);

            this.dirList =
                    (this.dirList == null)
                            ? DirPath.of().add(this.javacResult.classDir)
                            : this.dirList.add(this.javacResult.classDir);
            if (mavenStyleRoot != null) {
                var resources = mavenStyleRoot.dir("src/main/resources");
                if (resources.exists()) {
                    this.dirList.add(resources);
                }
            }
            return self();
        }

        public JavacBuilder javacBuilder(Consumer<JavacBuilder> javacBuilderConsumer) {
            JavacBuilder javacBuilder = new JavacBuilder(this);
            javacBuilderConsumer.accept(javacBuilder);
            return javacBuilder;
        }

        public JavacBuilder javacBuilder(JavacBuilder copyMe, Consumer<JavacBuilder> javacBuilderConsumer) {
            JavacBuilder javacBuilder = new JavacBuilder(this);
            javacBuilder.dontCallThisCopy(copyMe);
            javacBuilderConsumer.accept(javacBuilder);
            return javacBuilder;
        }

        public JarBuilder javac(Consumer<JavacBuilder> javacBuilderConsumer) {
            return javac(javacBuilder(javacBuilderConsumer));
        }

        public JarBuilder javac(JavacBuilder copyMe, Consumer<JavacBuilder> javacBuilderConsumer) {
            return javac(javacBuilder(copyMe, javacBuilderConsumer));
        }

        @SuppressWarnings("unchecked")
        public <P extends DirPathHolder<P>> JarBuilder dir_list(P... holders) {
            Arrays.asList(holders).forEach(holder ->
                    this.dirList = DirPath.ofOrUse(this.dirList).add(holder)
            );
            return self();
        }

        @SuppressWarnings("unchecked")
        public <P extends DirPathHolder<P>> JarBuilder add(P... holders) {
            Arrays.asList(holders).forEach(holder ->
                    this.dirList = DirPath.ofOrUse(this.dirList).add(holder)
            );
            return self();
        }
    }

    public static final class JarResult extends Result<JarBuilder> implements ClassPathEntryProvider {
        public StringList opts = new StringList();
        public List<RootDirAndSubPath> pathsToJar = new ArrayList<>();
        public List<Path> paths = new ArrayList<>();

        public JarResult(JarBuilder jarBuilder) {
            super(jarBuilder);
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(builder.jar);
        }

        @Override
        public String toString() {
            return builder.jar.path.toString();
        }
    }

    public static JarResult jar(JarBuilder jarBuilder) {

        JarResult result = new JarResult(jarBuilder);
        result.ok = true;
        try {

            var jarStream = new JarOutputStream(Files.newOutputStream(jarBuilder.jar.path()));
            if (jarBuilder.dirList == null) {
               result.ok=false;
            }
            if (result.ok) {
                if (jarBuilder.manifest != null) {
                    // We must add manifest
                    var entry = new JarEntry("META-INF/MANIFEST.MF");
                    // entry.setTime(Files.getLastModifiedTime(rootAndPath.path()).toMillis());
                    result.note(null, "Added manifest entry");
                    jarStream.putNextEntry(entry);
                    jarBuilder.manifest.writeTo(jarStream);
                    jarStream.closeEntry();

                }

                jarBuilder.dirList.entries.forEach(
                        root ->
                                root.findFiles()
                                        .map(path -> new RootDirAndSubPath(root, path))
                                        .forEach(result.pathsToJar::add));
                result.pathsToJar.stream()
                        .sorted(Comparator.comparing(RootDirAndSubPath::path))
                        .forEach(
                                rootAndPath -> {
                                    try {
                                        result.paths.add(rootAndPath.path);
                                        var entry = new JarEntry(rootAndPath.relativize().toString());
                                        entry.setTime(Files.getLastModifiedTime(rootAndPath.path()).toMillis());
                                        jarStream.putNextEntry(entry);
                                        Files.newInputStream(rootAndPath.path()).transferTo(jarStream);
                                        jarStream.closeEntry();
                                        if (jarBuilder.verbose) {
                                            println("INFO: adding " + rootAndPath.relativize().toString());
                                        }
                                    } catch (IOException e) {
                                        throw new RuntimeException(e);
                                    }
                                });
                jarStream.finish();
                jarStream.close();
                if (jarBuilder.verbose) {
                    println("INFO: created " + jarBuilder.jar.path.toString());
                }
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JarBuilder jarBuilder(Consumer<JarBuilder> jarBuilderConsumer) {
        JarBuilder jarBuilder = new JarBuilder();
        jarBuilderConsumer.accept(jarBuilder);
        return jarBuilder;
    }

    public static JarBuilder jarBuilder(JarBuilder copyMe, Consumer<JarBuilder> jarBuilderConsumer) {
        JarBuilder jarBuilder = new JarBuilder();
        jarBuilder.dontCallThisCopy(copyMe);
        jarBuilderConsumer.accept(jarBuilder);
        return jarBuilder;
    }

    public static JarResult jar(Consumer<JarBuilder> jarBuilderConsumer) {
        return jar(jarBuilder(jarBuilderConsumer));
    }

    public static JarResult jar(JarBuilder copyMe, Consumer<JarBuilder> jarBuilderConsumer) {
        return jar(jarBuilder(copyMe, jarBuilderConsumer));
    }

    public static final class CMakeBuilder extends Builder<CMakeBuilder> {
        public List<String> libraries = new ArrayList<>();
        public BuildDir cmakeBuildDir;
        public DirEntry sourceDir;
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

        public CMakeBuilder build_dir(BuildDir cmakeBuildDir) {
            this.cmakeBuildDir = cmakeBuildDir;
            opts("-B", cmakeBuildDir.path.toString());
            return this;
        }

        public CMakeBuilder copy_to(BuildDir copyToDir) {
            this.copyToDir = copyToDir;
            opts("-DHAT_TARGET=" + this.copyToDir.path().toString());
            return this;
        }

        public CMakeBuilder source_dir(DirEntry sourceDir) {
            this.sourceDir = sourceDir;
            opts("-S", sourceDir.path().toString());
            return this;
        }

        public CMakeBuilder build(BuildDir cmakeBuildDir) {
            this.cmakeBuildDir = cmakeBuildDir;
            opts("--build", cmakeBuildDir.path().toString());
            return this;
        }
        public CMakeBuilder target(String target) {
            opts("--target", target);
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

    public static Optional<Path> fromPATH(String name) {
        return Arrays.stream(System.getenv("PATH").split(File.pathSeparator))
                .map(dirName -> Path.of(dirName).resolve(name).normalize())
                .filter(Files::isExecutable)
                .filter(Files::isRegularFile)
                .findFirst();
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

    public record Regex(Pattern pattern) {
        Regex(String regex) {
            this(Pattern.compile(regex));
        }

        public static Regex of(String regexString) {
            return new Regex(regexString);
        }

        boolean matches(String text, Consumer<Matcher> matcherConsumer) {
            if (pattern().matcher(text) instanceof Matcher matcher && matcher.matches()) {
                matcherConsumer.accept(matcher);
                return true;
            } else {
                return false;
            }
        }

        boolean matches(String text) {
            return pattern().matcher(text) instanceof Matcher matcher && matcher.matches();
        }
    }


    public static class MavenStyleProject implements Script.ClassPathEntryProvider {

        final BuildDir buildDir;
        final Script.JarFile jarFile;
        final Script.DirEntry dir;
        final String name;
        JarResult result =null;
        final boolean hasJavaSources;

        final List<Script.ClassPathEntryProvider> classPath = new ArrayList<>();
        final List<Script.ClassPathEntryProvider> failedDependencies = new ArrayList<>();
        MavenStyleProject(BuildDir buildDir,Script.JarFile jarFile, Script.DirEntry dir, String name,  Script.ClassPathEntryProvider ...classPathEntryProviders) {
            this.buildDir = buildDir;
            this.jarFile = jarFile;
            this.dir = dir;
            this.name = name;
            this.classPath.addAll(List.of(classPathEntryProviders ));
            for (Script.ClassPathEntryProvider classPathEntryProvider : classPathEntryProviders) {
                classPathEntryProvider.classPathEntries().forEach(classPathEntry -> {
                    if (!classPathEntry.exists()){
                        failedDependencies.add(classPathEntry);
                    }
                });
            }
            this.hasJavaSources = dir.sourceDir("src/main/java").javaFiles().findAny().isPresent();
        }

        boolean canBeBuilt(){
            if (!failedDependencies.isEmpty()){
                print("Skipping "+jarFile.fileName()+" failed dependencies ");
                for (Script.ClassPathEntryProvider classPathEntryProvider : failedDependencies) {
                    classPathEntryProvider.classPathEntries().forEach(classPathEntry ->
                            print(classPathEntry.fileName())
                    );
                }
                println("");
                return false;
            }else if (!hasJavaSources) {
                println("Skipping " + jarFile.fileName() + " no java sources");
                return false;
            }
            return true;
        }



        public MavenStyleProject buildExcluding(Predicate<JavaSourceFile> sourceFilePredicate) {
            if (canBeBuilt()) {
                result = Script.jar(jar -> jar
                        .verbose(false)
                        .jarFile(jarFile)
                        .maven_style_root(dir)
                        .javac(javac -> javac
                                .enable_preview()
                                .add_modules("jdk.incubator.code")
                                .when(sourceFilePredicate != null, _ -> javac.exclude(sourceFilePredicate))
                                .current_source()
                                .class_path(classPath)
                        )
                );
              //  print(result.builder.javacResult);
                if ((result.ok && result.builder.javacResult.ok)){
                    println("Created \u001b[32m"+jarFile.fileName()+"\u001b[0m" );
                }else{
                    print(result.builder.javacResult);
                    println("Failed to create \u001b[31m"+jarFile.fileName()+"\u001b[0m" );
                }
            }
            return this;
        }
        public MavenStyleProject build() {
            return buildExcluding(null); // slight hack
        }

        @Override
        public List<Script.ClassPathEntry> classPathEntries() {
            return List.of(jarFile);
        }
    }

    static MavenStyleProject mavenStyleProject( BuildDir buildDir, Script.DirEntry dir, Script.JarFile jarFile,
                                                Script.ClassPathEntryProvider ...classPathEntryProviders) {
        return new MavenStyleProject(buildDir,jarFile, dir, dir.fileName(),classPathEntryProviders);

    }
}

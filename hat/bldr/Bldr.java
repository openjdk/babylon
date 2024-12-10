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

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;
import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.zip.ZipFile;

import static java.io.IO.print;
import static java.io.IO.println;

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


      //  public JarFile jarFile(String name, BiConsumer<JarBuilder, JarFile> biConsumer) {
           // var result = JarFile.of(path().resolve(name));
        //    return result.create(biConsumer).jarFile;
     //   }

      //  public JarFile jarFile(String name, Consumer<JarBuilder> consumer) {
           // var result = JarFile.of(path().resolve(name));
          //  return result.create(consumer).jarFile;
     //   }

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
    }

    public record JarFile(Path path) implements ClassPathEntry, FilePathHolder {
        public static JarFile of(Path path) {
            return new JarFile(path);
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
            super(path.toUri(), JavaFileObject.Kind.SOURCE);
            this.path=path;
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


    public abstract sealed static class Builder<T extends Builder<T>> permits  CMakeBuilder, FormatBuilder, JExtractBuilder, JarBuilder, JavaOpts, TestNGBuilder {
        public Builder<?> parent;
        public boolean verbose;
        public boolean quiet;
        @SuppressWarnings("unchecked")
        T self() {
            return (T) this;
        }

        public T copy(T other){
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

        Builder(Builder<?> parent){
            this.parent = parent;
        }
        Builder(){
            this(null);
        }
    }

    public abstract static sealed class Result<T extends Builder<T>> permits JExtractResult, JarResult, JavaResult, JavacResult {
        public boolean ok;
        public T builder;
        Result(T builder){
            this.builder = builder;
        }
    }

    public static class Strings {
        public List<String> strings = new ArrayList<>();
        Strings(){

        }
        Strings(Strings strings){
            add(strings);
        }
        Strings(List<String> strings){
            add(strings);
        }

        Strings(String ... strings){
            add(strings);
        }

        public Strings add(List<String> strings) {
            this.strings.addAll(strings);
            return this;
        }

        public Strings add(String... strings) {
            add(Arrays.asList(strings));
            return this;
        }

        public Strings add(Strings strings) {
            add(strings.strings);
            return this;
        }

        public String spaceSeperated() {
            StringBuilder stringBuilder = new StringBuilder();
            strings.forEach(opt->stringBuilder.append(stringBuilder.isEmpty()?"":" ").append(opt));
            return stringBuilder.toString();
        }
    }


    public static sealed  class JavaOpts<T extends JavaOpts<T>> extends Builder<T> {
        public Dir jdk = java.home;
        public Boolean enablePreview;
        public Strings modules ;
        record FromModulePackageToModule(String fromModule, String pkg, String toModule){}
        List<FromModulePackageToModule> exports;

        public T copy(T other){
            super.copy(other);
            if (other.jdk != null) {
                this.jdk = other.jdk;
            }
            if (other.enablePreview != null) {
                this.enablePreview=other.enablePreview;
            }
            if (other.modules != null) {
               this.modules = new Strings(other.modules);
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

        public T jdk(Dir jdk) {
            this.jdk = jdk;
            return self();
        }

        public T add_exports(String fromModule, String pkg, String toModule) {
             if (this.exports == null){
                 this.exports = new ArrayList<>();
             }
             exports.add(new FromModulePackageToModule(fromModule, pkg, toModule));
             return self();
        }

        public T add_modules(String... modules) {
            if (this.modules == null){
                this.modules = new Strings();
            }
            this.modules.add(modules);

            return self();
        }

        public T add_exports(String fromModule, List<String> packages, String toModule) {

            packages.forEach(p -> add_exports(fromModule, p, toModule));
            return self();
        }

        public T enable_preview() {
            this.enablePreview = true;
            return self();
        }



    }

    public abstract sealed static class JavaToolBuilder<T extends JavaToolBuilder<T>> extends JavaOpts<T> permits JavacBuilder,JavaBuilder{
        public ClassPath classPath;
        public T copy(T other){
            super.copy(other);
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
        public Dir mavenStyleRoot;
        public ClassDir classDir;
        public SourcePath sourcePath;
        public ClassPath modulePath;
        public SourcePath moduleSourcePath;
        public Integer source;
        public JavacBuilder copy(JavacBuilder other){
            super.copy(other);
            if (other.mavenStyleRoot != null){
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to maven style dir");
            }
            if (other.sourcePath != null){
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to a SourcePath");
            }
            if (other.moduleSourcePath != null){
                throw new RuntimeException("You are copying a JavacBuilder which is already bound to a ModuleSourcePath");
            }

            if (other.source !=null){
                this.source = other.source;
            }

            if (other.classPath != null){
                ClassPath.ofOrUse(this.classPath).add(other.classPath);
            }
            return this;
        }
        public JavacBuilder source(int version) {
             this.source = version;
             return self();
        }

        public JavacBuilder maven_style_root(Dir mavenStyleRoot) {
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

        public JavacBuilder module_source_path(List<SourcePathEntry> moduleSourcePathEntries) {
            this.moduleSourcePath = SourcePath.ofOrUse(this.moduleSourcePath).add(moduleSourcePathEntries);
            return this;
        }

        public JavacBuilder module_source_path(SourcePathEntry... moduleSourcePathEntries) {
            return module_source_path(List.of(moduleSourcePathEntries));
        }

        public JavacBuilder module_source_path(SourcePath moduleSourcePath) {
            return module_source_path(moduleSourcePath.entries());
        }

        public JavacBuilder(){
            super();
        }
        public JavacBuilder(JarBuilder jarBuilder) {
            super(jarBuilder);
        }
    }

    public static final class JavacResult extends Result<JavacBuilder>{
        Strings opts = new Strings();
        List<JavaSourceFile> sourceFiles = new ArrayList<>();
        List<JavaFileObject> classes = new ArrayList<>();
        ClassDir classDir;
        JavacResult(JavacBuilder builder) {
            super(builder);
        }
    }

    public static JavacResult javac(JavacBuilder javacBuilder) {
        JavacResult result = new JavacResult(javacBuilder);

        try {
            if (javacBuilder.source != null){
                result.opts.add("--source", javacBuilder.source.toString());
            }

            if (javacBuilder.enablePreview!=null && javacBuilder.enablePreview){
                result.opts.add("--enable-preview");
            }
            if (javacBuilder.modules!=null){
                javacBuilder.modules.strings.forEach(module->
                    result.opts.add("--add-modules", module)
                );
            }

            if (javacBuilder.exports!=null){
                javacBuilder.exports.forEach(fpt->{
                    result.opts.add("--add-exports=" + fpt.fromModule + "/" + fpt.pkg + "=" + fpt.toModule);
                });
            }

            result.classDir = javacBuilder.classDir == null ? ClassDir.temp() : javacBuilder.classDir;
            result.opts.add("-d", result.classDir.path().toString());
            if (javacBuilder.classPath != null) {
                result.opts.add("--class-path", javacBuilder.classPath.charSeparated());
            }else if (javacBuilder.modulePath != null){
            //https://dev.java/learn/modules/building/
                result.opts.add("--module-path", javacBuilder.modulePath.charSeparated());
            }else{
                println("Warning no class path or module path ");
                //throw new RuntimeException("No class path or module path provided");
            }
            var mavenStyleRoot =
                    ((javacBuilder.parent instanceof JarBuilder jarBuilder) && jarBuilder.mavenStyleRoot instanceof Dir fromJarBuilder)
                        ?fromJarBuilder
                        :javacBuilder.mavenStyleRoot;


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
            }else{
                var sourcePath = SourcePath.of().add(SourcePathEntry.of(mavenStyleRoot.path.resolve("src/main/java")));
                result.sourceFiles.addAll(sourcePath.javaFiles().map(JavaSourceFile::new).toList());
                result.opts.add("--source-path", sourcePath.charSeparated());

                if (javacBuilder.sourcePath != null && !javacBuilder.sourcePath.entries.isEmpty()){
                   throw new RuntimeException("You have specified --source-path AND provided maven_style_root ");
                }
            }

            DiagnosticListener<JavaFileObject> diagnosticListener =
                    (diagnostic) -> {
                        if (!diagnostic.getKind().equals(Diagnostic.Kind.NOTE)) {
                            System.out.println("javac "
                                    + diagnostic.getKind()
                                            + " "
                                            +((JavaSourceFile)(diagnostic.getSource())).path().toString()
                                            +"  "
                                            + diagnostic.getLineNumber()
                                            + ":"
                                            + diagnostic.getColumnNumber()
                                            + " "
                                            + diagnostic.getMessage(null));
                        }
                    };

            JavaCompiler javac = javax.tools.ToolProvider.getSystemJavaCompiler();
             if (javacBuilder.verbose  || javacBuilder.parent instanceof JarBuilder jarBuilder && jarBuilder.verbose) {
                println("javac "+result.opts.spaceSeperated());
            }
             JavaCompiler.CompilationTask compilationTask =
                    (javac.getTask(
                            new PrintWriter(System.err),
                            javac.getStandardFileManager(diagnosticListener, null, null),
                            diagnosticListener,
                            result.opts.strings,
                            null,
                            result.sourceFiles
                    ));
            ((com.sun.source.util.JavacTask) compilationTask).generate().forEach(javaFileObject -> {
                result.classes.add(javaFileObject);
            });
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    public static JavacBuilder javacBuilder(Consumer<JavacBuilder> javacBuilderConsumer){
        JavacBuilder javacBuilder= new JavacBuilder();
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
        public Strings args = new Strings();
        public Strings nativeAccessModules = new Strings();
        private boolean headless;

        public JavaBuilder enable_native_access(String module) {
            nativeAccessModules.add(module);
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
            this.startOnFirstThread =   true;
            return this;
        }

        public void headless() {
            this.headless = true;
        }
    }

    public static final class JavaResult extends Result<JavaBuilder>{
        Strings opts = new Strings();
        JavaResult(JavaBuilder javaBuilder) {
            super(javaBuilder);
        }
    }

    public static JavaBuilder java(JavaBuilder javaBuilder) {
        JavaResult result = new JavaResult(javaBuilder);
        result.opts.add(javaBuilder.jdk.path().resolve("bin/java").toString());
        if (javaBuilder.enablePreview != null && javaBuilder.enablePreview){
            result.opts.add("--enable-preview");
        }
        if (javaBuilder.modules!=null){
            javaBuilder.modules.strings.forEach(module->
                    result.opts.add("--add-modules", module)
            );
        }

        if (javaBuilder.exports!=null){
            javaBuilder.exports.forEach(fpt->{
                result.opts.add("--add-exports=" + fpt.fromModule + "/" + fpt.pkg + "=" + fpt.toModule);
            });
        }
        if (javaBuilder.headless) {
            result.opts.add("-Dheadless=true");
        }
        if (javaBuilder.startOnFirstThread){
            result.opts.add("-XstartOnFirstThread");
        }

        javaBuilder.nativeAccessModules.strings.forEach(module->
            result.opts.add("--enable-native-access=" + module)
        );

        if (javaBuilder.classPath != null) {
            result.opts.add("--class-path", javaBuilder.classPath.charSeparated());
        }
        if (javaBuilder.libraryPath != null) {
            result.opts.add("-Djava.library.path=" + javaBuilder.libraryPath.charSeparated());
        }
        result.opts.add(javaBuilder.mainClass);
        result.opts.add(javaBuilder.args.strings);

        try {
            var processBuilder = new ProcessBuilder().inheritIO().command(result.opts.strings);
            var process = processBuilder.start();
            if (javaBuilder.verbose) {
                println(result.opts.spaceSeperated());
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

    public static final class FormatBuilder extends Builder<FormatBuilder> {
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

    public static final class TestNGBuilder extends Builder<TestNGBuilder> {
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


        var testJarResult =
                jar(jar->jar
                        .jar(testNGBuilder.testJar)
                        .javac(javac->javac
                                .source(24)
                                                        .enable_preview()
                                                        .class_path(testNGBuilder.classPath, mavenJars)
                                                        .source_path(testNGBuilder.sourcePath)
                        )
                );

        java(
                $ ->
                        $.enable_preview()
                                .add_exports("java.base", "jdk.internal", "ALL-UNNAMED")
                                .enable_native_access("ALL-UNNAMED")
                                .class_path(testNGBuilder.classPath, mavenJars, testJarResult)
                                .main_class("org.testng.TestNG")
                                .args(testNGSuiteFile.path().toString()));
    }

    public static final class JarBuilder extends Builder<JarBuilder> {
        public Dir mavenStyleRoot;
        public JarFile jar;
        public JavacResult javacResult;
        public DirPath dirList;

        public JarBuilder jar(JarFile jar) {
            this.jar = jar;
            return self();
        }

        public JarBuilder maven_style_root(Dir mavenStyleRoot) {
            this.mavenStyleRoot = mavenStyleRoot;
            return this;
        }

        private JarBuilder javac(JavacBuilder javacBuilder) {
            this.javacResult = Bldr.javac(javacBuilder);

            this.dirList =
                    (this.dirList == null)
                            ? DirPath.of().add(this.javacResult.classDir)
                            : this.dirList.add(this.javacResult.classDir);
            if (mavenStyleRoot!=null){
                var resources = mavenStyleRoot.dir("src/main/resources");
                if (resources.exists()) {
                    this.dirList.add(resources);
                }
            }
            return self();
        }

        public JavacBuilder javacBuilder(Consumer<JavacBuilder> javacBuilderConsumer){
            JavacBuilder javacBuilder= new JavacBuilder(this);
            javacBuilderConsumer.accept(javacBuilder);
            return javacBuilder;
        }

        public JarBuilder javac(Consumer<JavacBuilder> javacBuilderConsumer) {
            return javac(javacBuilder(javacBuilderConsumer));
        }

        public <P extends DirPathHolder<P>> JarBuilder dir_list(P holder) {
            DirPath.ofOrUse(this.dirList).add(holder);
            return self();
        }
    }

    public static final class JarResult extends Result<JarBuilder> implements ClassPathEntryProvider{
        public Strings opts = new Strings();
        public List<RootDirAndSubPath> pathsToJar = new ArrayList<>();
        public List<Path> paths = new ArrayList<>();
        public JarFile jarFile;
        public JarResult(JarBuilder jarBuilder) {
            super(jarBuilder);
            this.jarFile = jarBuilder.jar;
        }

        @Override
        public List<ClassPathEntry> classPathEntries() {
            return List.of(jarFile);
        }

        @Override public String toString(){
            return jarFile.path.toString();
        }
    }

    public static JarResult jar(JarBuilder jarBuilder) {
        JarResult result = new JarResult(jarBuilder);
        try {

            var jarStream = new JarOutputStream(Files.newOutputStream(jarBuilder.jar.path()));
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
                                    if (jarBuilder.verbose){
                                        println("INFO: adding "+rootAndPath.relativize().toString());
                                    }
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            });
            jarStream.finish();
            jarStream.close();
            if (jarBuilder.verbose){
                println("INFO: created "+jarBuilder.jar.path.toString());
            }
            return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static JarBuilder jarBuilder(Consumer<JarBuilder> jarBuilderConsumer){
        JarBuilder jarBuilder= new JarBuilder();
        jarBuilderConsumer.accept(jarBuilder);
        return jarBuilder;
    }

    public static JarResult jar(Consumer<JarBuilder> jarBuilderConsumer) {
        return jar(jarBuilder(jarBuilderConsumer));
    }

    public static final class CMakeBuilder extends Builder<CMakeBuilder> {
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

    public static final class JExtractBuilder extends Builder<JExtractBuilder> {
        public Strings compileFlags = new Strings();
        public List<Path> libraries = new ArrayList<>();
        public List<Path> headers = new ArrayList<>();
        private String targetPackage;
        private BuildDir output;

        public JExtractBuilder copy(JExtractBuilder other){
            this.compileFlags = new Strings(other.compileFlags);
            if (other.targetPackage != null){
                throw new RuntimeException("You are copying jextract builder already bound to a target package");
            }
            if (other.output != null){
                throw new RuntimeException("You are copying jextract builder already bound to output directory");
            }
            if (!other.libraries.isEmpty()){
                throw new RuntimeException("You are copying jextract builder already bound to library(ies)");
            }
            if (!other.headers.isEmpty()){
                throw new RuntimeException("You are copying jextract builder already bound to headers library(ies)");
            }
            return self();
        }
        public JExtractBuilder target_package(String targetPackage) {
            this.targetPackage = targetPackage;
            return  self();
        }

        public JExtractBuilder output(BuildDir output) {
            this.output = output;
            return  self();
        }

        public JExtractBuilder library(Path... libraries) {
            this.libraries.addAll(Arrays.asList(libraries));
            return  self();
        }

        public JExtractBuilder compile_flag(String... compileFlags) {
            this.compileFlags.add(compileFlags);
            return  self();
        }

        public JExtractBuilder header(Path header) {
            this.headers.add(header);
            return  self();
        }
    }

    public static final class JExtractResult extends Result<JExtractBuilder>{
public Strings opts = new Strings();
        JExtractResult(JExtractBuilder builder) {
            super(builder);
        }
    }

    public static JExtractResult jextract(Jextract executable, Consumer<JExtractBuilder> jextractBuilderConsumer) {

        var exePath = executable.path;
        var homePath = exePath.getParent().getParent();

        JExtractBuilder jExtractBuilder = new JExtractBuilder();
        JExtractResult result = new JExtractResult(jExtractBuilder);
        jextractBuilderConsumer.accept(jExtractBuilder);
        result.opts.add(executable.path().toString());

        if (jExtractBuilder.targetPackage != null) {
            result.opts.add("--target-package", jExtractBuilder.targetPackage);
        }
        if (jExtractBuilder.output != null) {
            jExtractBuilder.output.create();
            result.opts.add("--output", jExtractBuilder.output.path().toString());
        }
        for (Path library : jExtractBuilder.libraries) {
            result.opts.add("--library", ":" + library);
        }

        for (Path header : jExtractBuilder.headers) {
            result.opts.add(header.toString());
        }

        if (jExtractBuilder.compileFlags != null && !jExtractBuilder.compileFlags.strings.isEmpty()) {
            jExtractBuilder.output.textFile("compile_flags.txt", jExtractBuilder.compileFlags.strings);
        }

        if (jExtractBuilder.verbose) {
            println(result.opts.spaceSeperated());
        }
        var processBuilder = new ProcessBuilder();
        if (jExtractBuilder.output != null) {
            processBuilder.directory(jExtractBuilder.output.path().toFile());
        }
        processBuilder.inheritIO().command(result.opts.strings);
        try {
            processBuilder.start().waitFor();
        } catch (InterruptedException | IOException ie) {
            throw new RuntimeException(ie);
        }
        return result;
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

    public static class CMakeProbe implements Capabilities.Probe {
        public interface CMakeVar<T> {
            String name();

            T value();
        }

        public record CMakeTypedVar(String name, String type, String value, String comment)
                implements CMakeVar<String> {
            static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+):([^=]*)=(.*)$");

            CMakeTypedVar(Matcher matcher, String comment) {
                this(
                        "CMAKE_" + matcher.group(1).trim(),
                        matcher.group(2).trim(),
                        matcher.group(3).trim(),
                        comment.substring(2).trim());
            }

            static boolean onMatch(String line, String comment, Consumer<CMakeTypedVar> consumer) {
                return regex.matches(line, matcher -> consumer.accept(new CMakeTypedVar(matcher, comment)));
            }
        }

        public record CMakeSimpleVar(String name, String value) implements CMakeVar {
            static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)\\}>\\}$");

            CMakeSimpleVar(Matcher matcher) {
                this(
                        "CMAKE_" + matcher.group(1).trim(),
                        (matcher.group(2).isEmpty()) ? "" : matcher.group(2).trim());
            }

            static boolean onMatch(String line, String comment, Consumer<CMakeSimpleVar> consumer) {
                return regex.matches(line, matcher -> consumer.accept(new CMakeSimpleVar(matcher)));
            }
        }

        public record CMakeDirVar(String name, DirPathHolder value) implements CMakeVar {
            static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)\\}>\\}$");

            static boolean onMatch(String line, String comment, Consumer<CMakeSimpleVar> consumer) {
                return regex.matches(line, matcher -> consumer.accept(new CMakeSimpleVar(matcher)));
            }
        }

        public record CMakeContentVar(String name, String value) implements CMakeVar {
            static final Regex startRegex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{(.*)$");
            static final Regex endRegex = Regex.of("^(.*)\\}>\\}$");
        }

        public record CMakeRecipeVar(String name, String value) implements CMakeVar<String> {
            static final Regex varPattern = Regex.of("<([^>]*)>");
            static final Regex regex = Regex.of("^_*(?:CMAKE_)?([A-Za-z0-9_]+)=\\{<\\{<(.*)>\\}>\\}$");

            CMakeRecipeVar(Matcher matcher) {
                this(
                        "CMAKE_" + matcher.group(1).trim(),
                        "<" + ((matcher.group(2).isEmpty()) ? "" : matcher.group(2).trim()) + ">");
            }

            public String expandRecursively(Map<String, CMakeVar<?>> varMap, String value) { // recurse
                String result = value;
                if (varPattern.pattern().matcher(value) instanceof Matcher matcher && matcher.find()) {
                    var v = matcher.group(1);
                    if (varMap.containsKey(v)) {
                        String replacement = varMap.get(v).value().toString();
                        result =
                                expandRecursively(
                                        varMap,
                                        value.substring(0, matcher.start())
                                                + replacement
                                                + value.substring(matcher.end()));
                    }
                }
                return result;
            }

            public String expand(Map<String, CMakeVar<?>> vars) {
                return expandRecursively(vars, value());
            }

            static boolean onMatch(String line, String comment, Consumer<CMakeRecipeVar> consumer) {
                return regex.matches(line, matcher -> consumer.accept(new CMakeRecipeVar(matcher)));
            }
        }

        BuildDir dir;

        Map<String, CMakeVar<?>> varMap = new HashMap<>();

        public CMakeProbe(BuildDir dir, Capabilities capabilities) {
            this.dir = BuildDir.of(dir.path("cmakeprobe"));
            this.dir.clean();

            try {
                this.dir.cmakeLists($-> {$
                        .append(
                             """
                             cmake_minimum_required(VERSION 3.21)
                             project(cmakeprobe)
                             set(CMAKE_CXX_STANDARD 14)
                             foreach(VarName ${VarNames})
                                message("${VarName}={<{${${VarName}}}>}")
                             endforeach()
                             """);
                            capabilities
                                    .capabilities()
                                    .filter(capability -> capability instanceof Capabilities.CMakeCapability)
                                    .map(capability -> (Capabilities.CMakeCapability) capability)
                                    .forEach(p -> $.append("find_package(").append(p.name).append(")\n")
                            );

                            //println("content = {"+$+"}");
                        });

                var cmakeProcessBuilder =
                        new ProcessBuilder()
                                .directory(this.dir.path().toFile())
                                .redirectErrorStream(true)
                                .command("cmake", "-LAH")
                                .start();
                List<String> lines =
                        new BufferedReader(new InputStreamReader(cmakeProcessBuilder.getInputStream()))
                                .lines()
                                .toList();

                String comment = null;
                String contentName = null;
                StringBuilder content = null;

                for (String line : lines) {

                 //   frameworkMap.values().forEach(framework ->
                   //     framework.regex.matches(line,
                    //            m->println(line)
                     //   )
                    //);
                    if (line.startsWith("//")) {
                        comment = line;
                        content = null;

                    } else if (comment != null) {
                        if (CMakeTypedVar.onMatch(
                                line,
                                comment,
                                v -> {
                                    if (varMap.containsKey(v.name())) {
                                        var theVar = varMap.get(v.name());
                                        if (theVar.value().equals(v.value())) {
                                          /*  println(
                                                    "replacing duplicate variable with typed variant with the name same value"
                                                            + v
                                                            + theVar);*/
                                        } else {
                                            throw new IllegalStateException(
                                                    "Duplicate variable name different value: " + v + theVar);
                                        }
                                        varMap.put(v.name(), v);
                                    } else {
                                        varMap.put(v.name(), v);
                                    }
                                })) {
                        } else {
                            println("failed to parse " + line);
                        }
                        comment = null;
                        content = null;
                        contentName = null;
                    } else if (!line.isEmpty()) {
                        if (content != null) {
                            if (CMakeContentVar.endRegex.pattern().matcher(line) instanceof Matcher matcher
                                    && matcher.matches()) {
                                content.append("\n").append(matcher.group(1));
                                var v = new CMakeContentVar(contentName, content.toString());
                                contentName = null;
                                content = null;
                                varMap.put(v.name(), v);
                            } else {
                                content.append("\n").append(line);
                            }
                        } else if (!line.endsWith("}>}")
                                && CMakeContentVar.startRegex.pattern().matcher(line) instanceof Matcher matcher
                                && matcher.matches()) {
                            contentName = "CMAKE_" + matcher.group(1);
                            content = new StringBuilder(matcher.group(2));
                        } else if (CMakeRecipeVar.regex.pattern().matcher(line) instanceof Matcher matcher
                                && matcher.matches()) {
                            CMakeVar<String> v = new CMakeRecipeVar(matcher);
                            if (varMap.containsKey(v.name())) {
                                var theVar = varMap.get(v.name());
                                if (theVar.value().equals(v.value())) {
                                  //  println("Skipping duplicate variable name different value: " + v + theVar);
                                } else {
                                    throw new IllegalStateException(
                                            "Duplicate variable name different value: " + v + theVar);
                                }
                                varMap.put(v.name(), v);
                            } else {
                                varMap.put(v.name(), v);
                            }
                        } else if (CMakeSimpleVar.regex.pattern().matcher(line) instanceof Matcher matcher
                                && matcher.matches()) {
                            var v =  new CMakeSimpleVar(matcher);
                            if (varMap.containsKey(v.name())) {
                                var theVar = varMap.get(v.name());
                                if (theVar.value().equals(v.value())) {
                                   // println("Skipping duplicate variable name different value: " + v + theVar);
                                } else {
                                    //throw new IllegalStateException(
                                      //      "Duplicate variable name different vars: " + v + theVar);
                                }
                                // note we don't replace a Typed with a Simple
                            } else {
                                varMap.put(v.name(), v);
                            }
                        } else {
                           // println("Skipping " + line);
                        }
                    }
                }

            } catch (IOException ioe) {
                throw new RuntimeException(ioe);
            }

            capabilities
                    .capabilities()
                    .filter(capability -> capability instanceof Capabilities.CMakeCapability)
                    .map(capability->(Capabilities.CMakeCapability)capability)
                    .forEach(capability -> capability.setCmakeProbe(this));

        }

        ObjectFile cxxCompileObject(
                ObjectFile target, CppSourceFile source, List<String> frameworks) {
            CMakeRecipeVar compileObject = (CMakeRecipeVar) varMap.get("CMAKE_CXX_COMPILE_OBJECT");
            Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
            localVars.put("DEFINES", new CMakeSimpleVar("DEFINES", ""));
            localVars.put("INCLUDES", new CMakeSimpleVar("INCLUDES", ""));
            localVars.put("FLAGS", new CMakeSimpleVar("FLAGS", ""));
            localVars.put("OBJECT", new CMakeSimpleVar("OBJECT", target.path().toString()));
            localVars.put("SOURCE", new CMakeSimpleVar("SOURCE", source.path().toString()));
            String executable = compileObject.expand(localVars);
            println(executable);
            return target;
        }

        ExecutableFile cxxLinkExecutable(
                ExecutableFile target, List<ObjectFile> objFiles, List<String> frameworks) {
            CMakeRecipeVar linkExecutable = (CMakeRecipeVar) varMap.get("CMAKE_CXX_LINK_EXECUTABLE");
            Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
            String executable = linkExecutable.expand(localVars);
            println(executable);
            return target;
        }

        SharedLibraryFile cxxCreateSharedLibrary(
                SharedLibraryFile target, List<ObjectFile> objFiles, List<String> frameworks) {
            CMakeRecipeVar createSharedLibrary =
                    (CMakeRecipeVar) varMap.get("CMAKE_CXX_CREATE_SHARED_LIBRARY");
            Map<String, CMakeVar<?>> localVars = new HashMap<>(varMap);
            String executable = createSharedLibrary.expand(localVars);
            println(executable);
            return target;
        }


        public String value(String key) {
            var  v = varMap.get(key);
            return v.value().toString();
        }

        public  boolean hasKey(String includeDirKey) {
            return varMap.containsKey(includeDirKey);
        }

    }

    public static class Capabilities {
        interface Probe{

        }
        public static abstract class Capability {
            final public String name;
            Capability(String name) {
                this.name=name;
            }
            public abstract boolean available();


        }
        public static abstract class CMakeCapability extends Capability {
            CMakeProbe cmakeProbe;
            CMakeCapability(String name) {
                super(name);
            }
            public  void setCmakeProbe(CMakeProbe cmakeProbe){
                this.cmakeProbe = cmakeProbe;
            }
        }

        public Map<String, Capability> capabilityMap = new HashMap<>();

        public static Capabilities of(Capability... capabilities) {
            return new Capabilities(capabilities);
        }

        public Stream<Capability> capabilities() {
            return capabilityMap.values().stream();
        }
        public Stream<Capability> capabilities(Predicate<Capability> filter) {
            return capabilities().filter(filter);
        }

        public boolean capabilityIsAvailable(String name) {
            return capabilities().anyMatch(c-> c.name.equalsIgnoreCase(name));
        }

        private Capabilities(Capability... capabilities){
            List.of(capabilities).forEach(capability ->
                    capabilityMap.put(capability.name, capability)
            );
        }

        public static class OpenCL extends CMakeCapability {
            public static String includeDirKey  = "CMAKE_OpenCL_INCLUDE_DIR";
            public static String libKey  = "CMAKE_OpenCL_LIBRARY";
            public static String osxSysroot = "CMAKE_OSX_SYSROOT";
            public OpenCL() {
                super("OpenCL");
            }
            public static OpenCL of(){
                return new OpenCL();
            }
            public String appLibFrameworks() {
                return cmakeProbe.value(osxSysroot);
            }

            @Override
            public boolean available() {
                return cmakeProbe.hasKey(includeDirKey);
            }
            public String lib(){
                return cmakeProbe.value(libKey);
            }

            public String includeDir(){
                return cmakeProbe.value(includeDirKey);
            }
        }

        public static class OpenGL extends CMakeCapability {
            public static String includeDirKey  = "CMAKE_OPENGL_INCLUDE_DIR";
            public OpenGL() {
                super("OpenGL");
            }
            public static OpenGL of(){
                return new OpenGL();
            }
            @Override
            public boolean available() {
                return cmakeProbe.hasKey(includeDirKey);
            }
            Dir includeDir(){
                return Dir.of(Path.of(cmakeProbe.value(includeDirKey)));
            }
        }

        public static class HIP extends CMakeCapability {
            public HIP() {
                super("HIP");
            }
            public static HIP of(){
                return new HIP();
            }
            @Override
            public boolean available() {
                return false;
            }
        }
        public static class CUDA extends CMakeCapability {
            public static String sdkRootDirKey  = "CMAKE_CUDA_SDK_ROOT_DIR";
            public static String sdkRootDirNotFoundValue  = "CUDA_SDK_ROOT_DIR-NOTFOUND";
            public CUDA() {
                super("CUDA");
            }
            public static CUDA of(){
                return new CUDA();
            }
            @Override
            public boolean available() {
                return cmakeProbe.hasKey(sdkRootDirKey) && !cmakeProbe.value(sdkRootDirKey).equals(sdkRootDirNotFoundValue);
            }
        }



    }

    static record Regex(Pattern pattern) {
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
    }

    public static class XMLNode {
        org.w3c.dom.Element element;
        List<XMLNode> children = new ArrayList<>();
        Map<String, String> attrMap = new HashMap<>();

        public static class AbstractXMLBuilder<T extends AbstractXMLBuilder<T>> {
            final public org.w3c.dom.Element element;
            @SuppressWarnings("unchecked")
            public T self() {
                return (T) this;
            }

            public T attr(String name, String value) {
                // var att = element.getOwnerDocument().createAttribute(name);
                // att.setValue(value);
                element.setAttribute(name, value);
                // element.appendChild(att);
                return self();
            }

            public T attr(URI uri, String name, String value) {
                // var att = element.getOwnerDocument().createAttribute(name);
                // att.setValue(value);
                element.setAttributeNS(uri.toString(), name, value);
                // element.appendChild(att);
                return self();
            }

            public T element(String name, Function<Element, T> factory, Consumer<T> xmlBuilderConsumer) {
                var node = element.getOwnerDocument().createElement(name);
                element.appendChild(node);
                var builder = factory.apply(node);
                xmlBuilderConsumer.accept(builder);
                return self();
            }

            public T element(
                    URI uri, String name, Function<Element, T> factory, Consumer<T> xmlBuilderConsumer) {
                var node = element.getOwnerDocument().createElementNS(uri.toString(), name);
                element.appendChild(node);
                var builder = factory.apply(node);
                xmlBuilderConsumer.accept(builder);
                return self();
            }

            AbstractXMLBuilder(Element element) {
                this.element = element;
            }

            public T text(String thisText) {
                var node = element.getOwnerDocument().createTextNode(thisText);
                element.appendChild(node);
                return self();
            }

            public T comment(String thisComment) {
                var node = element.getOwnerDocument().createComment(thisComment);
                element.appendChild(node);
                return self();
            }

            <L> T forEach(List<L> list, BiConsumer<T, L> biConsumer) {
                list.forEach(l -> biConsumer.accept(self(), l));
                return self();
            }

            <L> T forEach(Stream<L> stream, BiConsumer<T, L> biConsumer) {
                stream.forEach(l -> biConsumer.accept(self(), l));
                return self();
            }

            <L> T forEach(Stream<L> stream, Consumer<L> consumer) {
                stream.forEach(consumer);
                return self();
            }

            protected T then(Consumer<T> xmlBuilderConsumer) {
                xmlBuilderConsumer.accept(self());
                return self();
            }
        }

        public static class PomXmlBuilder extends AbstractXMLBuilder<bldr.Bldr.XMLNode.PomXmlBuilder> {
            PomXmlBuilder(Element element) {
                super(element);
            }

            public PomXmlBuilder element(String name, Consumer<PomXmlBuilder> xmlBuilderConsumer) {
                return element(name, PomXmlBuilder::new, xmlBuilderConsumer);
            }

            public PomXmlBuilder element(URI uri, String name, Consumer<PomXmlBuilder> xmlBuilderConsumer) {
                return element(uri, name, PomXmlBuilder::new, xmlBuilderConsumer);
            }

            public PomXmlBuilder modelVersion(String s) {
                return element("modelVersion", $ -> $.text(s));
            }

            public PomXmlBuilder pom(String groupId, String artifactId, String version) {
                return modelVersion("4.0.0").packaging("pom").ref(groupId, artifactId, version);
            }

            public PomXmlBuilder jar(String groupId, String artifactId, String version) {
                return modelVersion("4.0.0").packaging("jar").ref(groupId, artifactId, version);
            }

            public PomXmlBuilder groupId(String s) {
                return element("groupId", $ -> $.text(s));
            }

            public PomXmlBuilder artifactId(String s) {
                return element("artifactId", $ -> $.text(s));
            }

            public PomXmlBuilder packaging(String s) {
                return element("packaging", $ -> $.text(s));
            }

            public PomXmlBuilder version(String s) {
                return element("version", $ -> $.text(s));
            }

            public PomXmlBuilder build(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("build", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder plugins(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("plugins", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder plugin(
                    String groupId,
                    String artifactId,
                    String version,
                    Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element(
                        "plugin", $ -> $.ref(groupId, artifactId, version).then(pomXmlBuilderConsumer));
            }

            public PomXmlBuilder antPlugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return plugin(
                        "org.apache.maven.plugins",
                        "maven-antrun-plugin",
                        "1.8",
                        pomXmlBuilderConsumer);
            }
            public PomXmlBuilder surefirePlugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return plugin(
                        "org.apache.maven.plugins",
                        "maven-surefire-plugin",
                        "3.1.2",
                        pomXmlBuilderConsumer);
            }

            public PomXmlBuilder compilerPlugin(
                    Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return plugin(
                        "org.apache.maven.plugins",
                        "maven-compiler-plugin",
                        "3.11.0",pomXmlBuilderConsumer
                      );
            }

            public PomXmlBuilder execPlugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return plugin("org.codehaus.mojo", "exec-maven-plugin", "3.1.0", pomXmlBuilderConsumer);
            }


            public PomXmlBuilder plugin(
                    String groupId, String artifactId, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("plugin", $ -> $.groupIdArtifactId(groupId, artifactId).then(pomXmlBuilderConsumer));
            }

            public PomXmlBuilder plugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("plugin", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder parent(String groupId, String artifactId, String version) {
                return parent(parent -> parent.ref(groupId, artifactId, version));
            }

            public PomXmlBuilder parent(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("parent", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder pluginManagement(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("pluginManagement", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder file(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("file", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder activation(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("activation", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder profiles(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("profiles", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder profile(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("profile", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder arguments(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("arguments", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder executions(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("executions", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder execution(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("execution", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder execIdPhaseConf(
                    String id, String phase, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return execution(execution -> execution.id(id).phase(phase).goals(gs -> gs.goal("exec")).configuration(pomXmlBuilderConsumer));
            }

            public PomXmlBuilder exec(
                    String phase, String executable, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return execIdPhaseConf(
                        executable + "-" + phase,
                        phase,
                        conf -> conf.executable(executable).arguments(pomXmlBuilderConsumer));
            }

            public PomXmlBuilder cmake(
                    String id, String phase, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return execIdPhaseConf(
                        id, phase, conf -> conf.executable("cmake").arguments(pomXmlBuilderConsumer));
            }

            public PomXmlBuilder cmake(String id, String phase, String... args) {
                return execIdPhaseConf(
                        id,
                        phase,
                        conf ->
                                conf.executable("cmake")
                                        .arguments(arguments -> arguments.forEach(Stream.of(args), arguments::argument)));
            }

            public PomXmlBuilder jextract(String id, String phase, String... args) {
                return execIdPhaseConf(
                        id,
                        phase,
                        conf ->
                                conf.executable("jextract")
                                        .arguments(arguments -> arguments.forEach(Stream.of(args), arguments::argument)));
            }

            public PomXmlBuilder ant(
                    String id, String phase, String goal, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return execution(execution -> execution
                                        .id(id)
                                        .phase(phase)
                                        .goals(gs -> gs.goal(goal))
                                        .configuration(configuration -> configuration.target(pomXmlBuilderConsumer)));
            }

            public PomXmlBuilder goals(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("goals", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder target(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("target", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder configuration(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("configuration", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder compilerArgs(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("compilerArgs", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder compilerArgs(String... args) {
                return element("compilerArgs", $ -> $.forEach(Stream.of(args), $::arg));
            }

            public PomXmlBuilder properties(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("properties", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder dependencies(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("dependencies", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder dependsOn(String groupId, String artifactId, String version) {
                return element("dependencies", $ -> $.dependency(groupId, artifactId, version));
            }
            public PomXmlBuilder dependsOn(String groupId, String artifactId, String version, String phase) {
                return element("dependencies", $ -> $.dependency(groupId, artifactId, version, phase));
            }

            public PomXmlBuilder dependency(String groupId, String artifactId, String version) {
                return dependency($ -> $.ref(groupId, artifactId, version));
            }

            public PomXmlBuilder dependency(
                    String groupId, String artifactId, String version, String scope) {
                return dependency($ -> $.ref(groupId, artifactId, version).scope(scope));
            }

            public PomXmlBuilder dependency(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("dependency", pomXmlBuilderConsumer);
            }

            public PomXmlBuilder modules(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
                return element("modules", pomXmlBuilderConsumer);
            }
            public PomXmlBuilder modules(List<String> modules) {
                return element("modules", $ -> $.forEach(modules.stream(), $::module));
            }
            public PomXmlBuilder modules(String... modules) {
                return modules(List.of(modules));
            }

            public PomXmlBuilder module(String name) {
                return element("module", $ -> $.text(name));
            }

            public PomXmlBuilder property(String name, String value) {
                return element(name, $ -> $.text(value));
            }
            public PomXmlBuilder antproperty(String name, String value) {
                return element("property", $ -> $.attr("name", name).attr("value", value));
            }

            public PomXmlBuilder scope(String s) {
                return element("scope", $ -> $.text(s));
            }

            public PomXmlBuilder phase(String s) {
                return element("phase", $ -> $.text(s));
            }

            public PomXmlBuilder argument(String s) {
                return element("argument", $ -> $.text(s));
            }

            public PomXmlBuilder goal(String s) {
                return element("goal", $ -> $.text(s));
            }

            public PomXmlBuilder copy(String file, String toDir) {
                return element("copy", $ -> $.attr("file", file).attr("toDir", toDir));
            }

            public PomXmlBuilder echo(String message) {
                return element("echo", $ -> $.attr("message", message));
            }

            public PomXmlBuilder echo(String filename, String message) {
                return element("echo", $ -> $.attr("message", message).attr("file", filename));
            }

            public PomXmlBuilder mkdir(String dirName) {
                return element("mkdir", $ -> $.attr("dir", dirName));
            }

            public PomXmlBuilder groupIdArtifactId(String groupId, String artifactId) {
                return groupId(groupId).artifactId(artifactId);
            }

            public PomXmlBuilder ref(String groupId, String artifactId, String version) {
                return groupIdArtifactId(groupId, artifactId).version(version);
            }

            public PomXmlBuilder skip(String string) {
                return element("skip", $ -> $.text(string));
            }

            public PomXmlBuilder id(String s) {
                return element("id", $ -> $.text(s));
            }

            public PomXmlBuilder arg(String s) {
                return element("arg", $ -> $.text(s));
            }

            public PomXmlBuilder argLine(String s) {
                return element("argLine", $ -> $.text(s));
            }

            public PomXmlBuilder source(String s) {
                return element("source", $ -> $.text(s));
            }

            public PomXmlBuilder target(String s) {
                return element("target", $ -> $.text(s));
            }

            public PomXmlBuilder showWarnings(String s) {
                return element("showWarnings", $ -> $.text(s));
            }

            public PomXmlBuilder showDeprecation(String s) {
                return element("showDeprecation", $ -> $.text(s));
            }

            public PomXmlBuilder failOnError(String s) {
                return element("failOnError", $ -> $.text(s));
            }

            public PomXmlBuilder exists(String s) {
                return element("exists", $ -> $.text(s));
            }

            public PomXmlBuilder activeByDefault(String s) {
                return element("activeByDefault", $ -> $.text(s));
            }

            public PomXmlBuilder executable(String s) {
                return element("executable", $ -> $.text(s));
            }

            public PomXmlBuilder workingDirectory(String s) {
                return element("workingDirectory", $ -> $.text(s));
            }
        }

        public static class ImlBuilder extends AbstractXMLBuilder<bldr.Bldr.XMLNode.ImlBuilder> {

            ImlBuilder(Element element) {
                super(element);
            }

            public ImlBuilder element(String name, Consumer<ImlBuilder> xmlBuilderConsumer) {
                return element(name, ImlBuilder::new, xmlBuilderConsumer);
            }

            public ImlBuilder element(URI uri, String name, Consumer<ImlBuilder> xmlBuilderConsumer) {
                return element(uri, name, ImlBuilder::new, xmlBuilderConsumer);
            }

            public ImlBuilder modelVersion(String s) {
                return element("modelVersion", $ -> $.text(s));
            }

            public ImlBuilder groupId(String s) {
                return element("groupId", $ -> $.text(s));
            }

            public ImlBuilder artifactId(String s) {
                return element("artifactId", $ -> $.text(s));
            }

            public ImlBuilder packaging(String s) {
                return element("packaging", $ -> $.text(s));
            }

            public ImlBuilder version(String s) {
                return element("version", $ -> $.text(s));
            }

            public ImlBuilder build(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("build", pomXmlBuilderConsumer);
            }

            public ImlBuilder plugins(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("plugins", pomXmlBuilderConsumer);
            }

            public ImlBuilder plugin(
                    String groupId,
                    String artifactId,
                    String version,
                    Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element(
                        "plugin",
                        $ ->
                                $.groupIdArtifactIdVersion(groupId, artifactId, version).then(pomXmlBuilderConsumer));
            }

            public ImlBuilder plugin(
                    String groupId, String artifactId, Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element(
                        "plugin", $ -> $.groupIdArtifactId(groupId, artifactId).then(pomXmlBuilderConsumer));
            }

            public ImlBuilder plugin(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("plugin", pomXmlBuilderConsumer);
            }

            public ImlBuilder parent(String groupId, String artifactId, String version) {
                return parent(parent -> parent.groupIdArtifactIdVersion(groupId, artifactId, version));
            }

            public ImlBuilder parent(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("parent", pomXmlBuilderConsumer);
            }

            public ImlBuilder pluginManagement(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("pluginManagement", pomXmlBuilderConsumer);
            }

            public ImlBuilder file(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("file", pomXmlBuilderConsumer);
            }

            public ImlBuilder activation(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("activation", pomXmlBuilderConsumer);
            }

            public ImlBuilder profiles(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("profiles", pomXmlBuilderConsumer);
            }

            public ImlBuilder profile(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("profile", pomXmlBuilderConsumer);
            }

            public ImlBuilder arguments(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("arguments", pomXmlBuilderConsumer);
            }

            public ImlBuilder executions(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("executions", pomXmlBuilderConsumer);
            }

            public ImlBuilder execution(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("execution", pomXmlBuilderConsumer);
            }

            public ImlBuilder goals(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("goals", pomXmlBuilderConsumer);
            }

            public ImlBuilder target(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("target", pomXmlBuilderConsumer);
            }

            public ImlBuilder configuration(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("configuration", pomXmlBuilderConsumer);
            }

            public ImlBuilder compilerArgs(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("compilerArgs", pomXmlBuilderConsumer);
            }

            public ImlBuilder properties(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("properties", pomXmlBuilderConsumer);
            }

            public ImlBuilder dependencies(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("dependencies", pomXmlBuilderConsumer);
            }

            public ImlBuilder dependency(String groupId, String artifactId, String version) {
                return dependency($ -> $.groupIdArtifactIdVersion(groupId, artifactId, version));
            }

            public ImlBuilder dependency(String groupId, String artifactId, String version, String scope) {
                return dependency($ -> $.groupIdArtifactIdVersion(groupId, artifactId, version).scope(scope));
            }

            public ImlBuilder dependency(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("dependency", pomXmlBuilderConsumer);
            }

            public ImlBuilder modules(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
                return element("modules", pomXmlBuilderConsumer);
            }

            public ImlBuilder module(String name) {
                return element("module", $ -> $.text(name));
            }

            public ImlBuilder property(String name, String value) {
                return element(name, $ -> $.text(value));
            }

            public ImlBuilder scope(String s) {
                return element("scope", $ -> $.text(s));
            }

            public ImlBuilder phase(String s) {
                return element("phase", $ -> $.text(s));
            }

            public ImlBuilder argument(String s) {
                return element("argument", $ -> $.text(s));
            }

            public ImlBuilder goal(String s) {
                return element("goal", $ -> $.text(s));
            }

            public ImlBuilder copy(String file, String toDir) {
                return element("copy", $ -> $.attr("file", file).attr("toDir", toDir));
            }

            public ImlBuilder groupIdArtifactId(String groupId, String artifactId) {
                return groupId(groupId).artifactId(artifactId);
            }

            public ImlBuilder groupIdArtifactIdVersion(String groupId, String artifactId, String version) {
                return groupIdArtifactId(groupId, artifactId).version(version);
            }

            public ImlBuilder skip(String string) {
                return element("skip", $ -> $.text(string));
            }

            public ImlBuilder id(String s) {
                return element("id", $ -> $.text(s));
            }

            public ImlBuilder arg(String s) {
                return element("arg", $ -> $.text(s));
            }

            public ImlBuilder argLine(String s) {
                return element("argLine", $ -> $.text(s));
            }

            public ImlBuilder source(String s) {
                return element("source", $ -> $.text(s));
            }

            public ImlBuilder target(String s) {
                return element("target", $ -> $.text(s));
            }

            public ImlBuilder showWarnings(String s) {
                return element("showWarnings", $ -> $.text(s));
            }

            public ImlBuilder showDeprecation(String s) {
                return element("showDeprecation", $ -> $.text(s));
            }

            public ImlBuilder failOnError(String s) {
                return element("failOnError", $ -> $.text(s));
            }

            public ImlBuilder exists(String s) {
                return element("exists", $ -> $.text(s));
            }

            public ImlBuilder activeByDefault(String s) {
                return element("activeByDefault", $ -> $.text(s));
            }

            public ImlBuilder executable(String s) {
                return element("executable", $ -> $.text(s));
            }
        }

        public static class XMLBuilder extends AbstractXMLBuilder<bldr.Bldr.XMLNode.XMLBuilder> {
           XMLBuilder(Element element) {
                super(element);
            }

            public XMLBuilder element(String name, Consumer<XMLBuilder> xmlBuilderConsumer) {
                return element(name, XMLBuilder::new, xmlBuilderConsumer);
            }

            public XMLBuilder element(URI uri, String name, Consumer<XMLBuilder> xmlBuilderConsumer) {
                return element(uri, name, XMLBuilder::new, xmlBuilderConsumer);
            }
        }

        static XMLNode create(String nodeName, Consumer<XMLBuilder> xmlBuilderConsumer) {

            try {
                var doc =
                        javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
                var element = doc.createElement(nodeName);
                doc.appendChild(element);
                XMLBuilder xmlBuilder = new XMLBuilder(element);
                xmlBuilderConsumer.accept(xmlBuilder);
                return new XMLNode(element);
            } catch (ParserConfigurationException e) {
                throw new RuntimeException(e);
            }
        }

        static XMLNode createIml(String commentText, Consumer<ImlBuilder> imlBuilderConsumer) {
            try {
                var doc =
                        javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
                var uri1 = URI.create("http://maven.apache.org/POM/4.0.0");
                var uri2 = URI.create("http://www.w3.org/2001/XMLSchema-instance");
                var uri3 = URI.create("http://maven.apache.org/xsd/maven-4.0.0.xsd");
                var comment = doc.createComment(commentText);
                doc.appendChild(comment);
                var element = doc.createElementNS(uri1.toString(), "project");
                doc.appendChild(element);
                element.setAttributeNS(uri2.toString(), "xsi:schemaLocation", uri1 + " " + uri3);
                ImlBuilder imlBuilder = new ImlBuilder(element);
                imlBuilderConsumer.accept(imlBuilder);
                return new XMLNode(element);
            } catch (ParserConfigurationException e) {
                throw new RuntimeException(e);
            }
        }

        public static XMLNode createPom(
                String commentText, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
            try {
                var doc =
                        javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

                var uri1 = URI.create("http://maven.apache.org/POM/4.0.0");
                var uri2 = URI.create("http://www.w3.org/2001/XMLSchema-instance");
                var uri3 = URI.create("http://maven.apache.org/xsd/maven-4.0.0.xsd");
                var comment = doc.createComment(commentText);
                doc.appendChild(comment);
                var element = doc.createElementNS(uri1.toString(), "project");
                doc.appendChild(element);
                element.setAttributeNS(uri2.toString(), "xsi:schemaLocation", uri1 + " " + uri3);
                PomXmlBuilder pomXmlBuilder = new PomXmlBuilder(element);
                pomXmlBuilderConsumer.accept(pomXmlBuilder);
                return new XMLNode(element);
            } catch (ParserConfigurationException e) {
                throw new RuntimeException(e);
            }
        }

        static XMLNode create(URI uri, String nodeName, Consumer<XMLBuilder> xmlBuilderConsumer) {
            try {
                var doc =
                        javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
                var element = doc.createElementNS(uri.toString(), nodeName);
                doc.appendChild(element);
                XMLBuilder xmlBuilder = new XMLBuilder(element);
                xmlBuilderConsumer.accept(xmlBuilder);
                return new XMLNode(element);
            } catch (ParserConfigurationException e) {
                throw new RuntimeException(e);
            }
        }

        XMLNode(Element element) {
            this.element = element;
            this.element.normalize();
            NodeList nodeList = element.getChildNodes();
            for (int i = 0; i < nodeList.getLength(); i++) {
                if (nodeList.item(i) instanceof Element e) {
                    this.children.add(new XMLNode(e));
                }
            }
            for (int i = 0; i < element.getAttributes().getLength(); i++) {
                if (element.getAttributes().item(i) instanceof org.w3c.dom.Attr attr) {
                    this.attrMap.put(attr.getName(), attr.getValue());
                }
            }
        }

        public boolean hasAttr(String name) {
            return attrMap.containsKey(name);
        }

        public String attr(String name) {
            return attrMap.get(name);
        }

        static Document parse(InputStream is) {
            try {
                return javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(is);
            } catch (ParserConfigurationException | SAXException | IOException e) {
                throw new RuntimeException(e);
            }
        }

        static Document parse(Path path) {
            try {
                return parse(Files.newInputStream(path));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        XMLNode(Path path) {
            this(parse(path).getDocumentElement());
        }

        XMLNode(File file) {
            this(parse(file.toPath()).getDocumentElement());
        }

        XMLNode(URL url) throws Throwable {
            this(parse(url.openStream()).getDocumentElement());
        }

        void write(StreamResult streamResult) throws Throwable {
            var transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty(OutputKeys.METHOD, "xml");
            transformer.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "no");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
            transformer.transform(new DOMSource(element.getOwnerDocument()), streamResult);
        }

        void write(File file) {
            try {
                write(new StreamResult(file));
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }

        public void write(XMLFile xmlFile) {
            try {
                write(new StreamResult(xmlFile.path().toFile()));
            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }

        @Override
        public String toString() {
            var stringWriter = new StringWriter();
            try {
                var transformer = TransformerFactory.newInstance().newTransformer();
                transformer.setOutputProperty(OutputKeys.INDENT, "yes");
                transformer.setOutputProperty(OutputKeys.METHOD, "xml");
                transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
                transformer.transform(new DOMSource(element), new StreamResult(stringWriter));
                return stringWriter.toString();
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }

        XPathExpression xpath(String expression) {
            XPath xpath = XPathFactory.newInstance().newXPath();
            try {
                return xpath.compile(expression);
            } catch (XPathExpressionException e) {
                throw new RuntimeException(e);
            }
        }

        Node node(XPathExpression xPathExpression) {
            try {
                return (Node) xPathExpression.evaluate(this.element, XPathConstants.NODE);
            } catch (XPathExpressionException e) {
                throw new RuntimeException(e);
            }
        }

        Optional<Node> optionalNode(XPathExpression xPathExpression) {
            var nodes = nodes(xPathExpression).toList();
            return switch (nodes.size()) {
                case 0 -> Optional.empty();
                case 1 -> Optional.of(nodes.getFirst());
                default -> throw new IllegalStateException("Expected 0 or 1 but got more");
            };
        }

        String str(XPathExpression xPathExpression) {
            try {
                return (String) xPathExpression.evaluate(this.element, XPathConstants.STRING);
            } catch (XPathExpressionException e) {
                throw new RuntimeException(e);
            }
        }

        String xpathQueryString(String xpathString) {
            try {
                return (String) xpath(xpathString).evaluate(this.element, XPathConstants.STRING);
            } catch (XPathExpressionException e) {
                throw new RuntimeException(e);
            }
        }

        NodeList nodeList(XPathExpression xPathExpression) {
            try {
                return (NodeList) xPathExpression.evaluate(this.element, XPathConstants.NODESET);
            } catch (XPathExpressionException e) {
                throw new RuntimeException(e);
            }
        }

        Stream<Node> nodes(XPathExpression xPathExpression) {
            var nodeList = nodeList(xPathExpression);
            List<Node> nodes = new ArrayList<>();
            for (int i = 0; i < nodeList.getLength(); i++) {
                nodes.add(nodeList.item(i));
            }
            return nodes.stream();
        }

        Stream<Element> elements(XPathExpression xPathExpression) {
            return nodes(xPathExpression)
                    .filter(n -> n instanceof Element)
                    .map(n -> (Element) n);
        }

        Stream<XMLNode> xmlNodes(XPathExpression xPathExpression) {
            return elements(xPathExpression).map(e -> new XMLNode(e));
        }
    }

    public static class MavenStyleRepository {
        private final String repoBase = "https://repo1.maven.org/maven2/";
        private final String searchBase = "https://search.maven.org/solrsearch/";
        public RepoDir dir;

        JarFile jarFile(Id id) {
            return dir.jarFile(id.artifactAndVersion() + ".jar");
        }

        XMLFile pomFile(Id id) {
            return dir.xmlFile(id.artifactAndVersion() + ".pom");
        }

        public enum Scope {
            TEST,
            COMPILE,
            PROVIDED,
            RUNTIME,
            SYSTEM;

            static Scope of(String name) {
                return switch (name.toLowerCase()) {
                    case "test" -> TEST;
                    case "compile" -> COMPILE;
                    case "provided" -> PROVIDED;
                    case "runtime" -> RUNTIME;
                    case "system" -> SYSTEM;
                    default -> COMPILE;
                };
            }
        }

        public record GroupAndArtifactId(GroupId groupId, ArtifactId artifactId) {

            public static GroupAndArtifactId of(String groupAndArtifactId) {
                int idx = groupAndArtifactId.indexOf('/');
                return of(groupAndArtifactId.substring(0, idx), groupAndArtifactId.substring(idx + 1));
            }

            public static GroupAndArtifactId of(GroupId groupId, ArtifactId artifactId) {
                return new GroupAndArtifactId(groupId, artifactId);
            }

            public static GroupAndArtifactId of(String groupId, String artifactId) {
                return of(GroupId.of(groupId), ArtifactId.of(artifactId));
            }

            String location() {
                return groupId().string().replace('.', '/') + "/" + artifactId().string();
            }

            @Override
            public String toString() {
                return groupId() + "/" + artifactId();
            }
        }

        public sealed interface Id permits DependencyId, bldr.Bldr.MavenStyleRepository.MetaDataId {
            MavenStyleRepository mavenStyleRepository();

            GroupAndArtifactId groupAndArtifactId();

            VersionId versionId();

            default String artifactAndVersion() {
                return groupAndArtifactId().artifactId().string() + '-' + versionId();
            }

            default String location() {
                return mavenStyleRepository().repoBase + groupAndArtifactId().location() + "/" + versionId();
            }

            default URL url(String suffix) {
                try {
                    return new URI(location() + "/" + artifactAndVersion() + "." + suffix).toURL();
                } catch (MalformedURLException | URISyntaxException e) {
                    throw new RuntimeException(e);
                }
            }
        }

        public record DependencyId(
                MavenStyleRepository mavenStyleRepository,
                GroupAndArtifactId groupAndArtifactId,
                VersionId versionId,
                Scope scope,
                boolean required)
                implements Id {
            @Override
            public String toString() {
                return groupAndArtifactId().toString()
                        + "/"
                        + versionId()
                        + ":"
                        + scope.toString()
                        + ":"
                        + (required ? "Required" : "Optiona");
            }
        }

        public record Pom(MetaDataId metaDataId, XMLNode xmlNode) {
            JarFile getJar() {
                var jarFile = metaDataId.mavenStyleRepository().jarFile(metaDataId); // ;
                metaDataId.mavenStyleRepository.queryAndCache(metaDataId.jarURL(), jarFile);
                return jarFile;
            }

            String description() {
                return xmlNode().xpathQueryString("/project/description/text()");
            }

            Stream<DependencyId> dependencies() {
                return xmlNode()
                        .nodes(xmlNode.xpath("/project/dependencies/dependency"))
                        .map(node -> new XMLNode((Element) node))
                        .map(
                                dependency ->
                                        new DependencyId(
                                                metaDataId().mavenStyleRepository(),
                                                bldr.Bldr.MavenStyleRepository.GroupAndArtifactId.of(
                                                        bldr.Bldr.MavenStyleRepository.GroupId.of(dependency.xpathQueryString("groupId/text()")),
                                                        bldr.Bldr.MavenStyleRepository.ArtifactId.of(dependency.xpathQueryString("artifactId/text()"))),
                                                bldr.Bldr.MavenStyleRepository.VersionId.of(dependency.xpathQueryString("version/text()")),
                                                bldr.Bldr.MavenStyleRepository.Scope.of(dependency.xpathQueryString("scope/text()")),
                                                !Boolean.parseBoolean(dependency.xpathQueryString("optional/text()"))));
            }

            Stream<DependencyId> requiredDependencies() {
                return dependencies().filter(DependencyId::required);
            }
        }

        public Optional<Pom> pom(Id id) {
            return switch (id) {
                case MetaDataId metaDataId -> {
                    if (metaDataId.versionId() == VersionId.UNSPECIFIED) {
                        // println("what to do when the version is unspecified");
                        yield Optional.empty();
                    }
                    try {
                        yield Optional.of(
                                new Pom(
                                        metaDataId,
                                        queryAndCache(
                                                metaDataId.pomURL(), metaDataId.mavenStyleRepository.pomFile(metaDataId))));
                    } catch (Throwable e) {
                        throw new RuntimeException(e);
                    }
                }
                case DependencyId dependencyId -> {
                    if (metaData(
                            id.groupAndArtifactId().groupId().string(),
                            id.groupAndArtifactId().artifactId().string())
                            instanceof Optional<MetaData> optionalMetaData
                            && optionalMetaData.isPresent()) {
                        if (optionalMetaData
                                .get()
                                .metaDataIds()
                                .filter(metaDataId -> metaDataId.versionId().equals(id.versionId()))
                                .findFirst()
                                instanceof Optional<MetaDataId> metaId
                                && metaId.isPresent()) {
                            yield pom(metaId.get());
                        } else {
                            yield Optional.empty();
                        }
                    } else {
                        yield Optional.empty();
                    }
                }
                default -> throw new IllegalStateException("Unexpected value: " + id);
            };
        }

        public Optional<Pom> pom(GroupAndArtifactId groupAndArtifactId) {
            var metaData = metaData(groupAndArtifactId).orElseThrow();
            var metaDataId = metaData.latestMetaDataId().orElseThrow();
            return pom(metaDataId);
        }

        record IdVersions(GroupAndArtifactId groupAndArtifactId, Set<Id> versions) {
            static IdVersions of(GroupAndArtifactId groupAndArtifactId) {
                return new IdVersions(groupAndArtifactId, new HashSet<>());
            }
        }

        public static class Dag implements ClassPathEntryProvider {
            private final MavenStyleRepository repo;
            private final List<GroupAndArtifactId> rootGroupAndArtifactIds;
            Map<GroupAndArtifactId, IdVersions> nodes = new HashMap<>();
            Map<IdVersions, List<IdVersions>> edges = new HashMap<>();

            Dag add(Id from, Id to) {
                var fromNode =
                        nodes.computeIfAbsent(
                                from.groupAndArtifactId(), _ -> IdVersions.of(from.groupAndArtifactId()));
                fromNode.versions().add(from);
                var toNode =
                        nodes.computeIfAbsent(
                                to.groupAndArtifactId(), _ -> IdVersions.of(to.groupAndArtifactId()));
                toNode.versions().add(to);
                edges.computeIfAbsent(fromNode, k -> new ArrayList<>()).add(toNode);
                return this;
            }

            void removeUNSPECIFIED() {
                nodes
                        .values()
                        .forEach(
                                idversions -> {
                                    if (idversions.versions().size() > 1) {
                                        List<Id> versions = new ArrayList<>(idversions.versions());
                                        idversions.versions().clear();
                                        idversions
                                                .versions()
                                                .addAll(
                                                        versions.stream()
                                                                .filter(v -> !v.versionId().equals(VersionId.UNSPECIFIED))
                                                                .toList());
                                        println(idversions);
                                    }
                                    if (idversions.versions().size() > 1) {
                                        throw new IllegalStateException("more than one version");
                                    }
                                });
            }

            Dag(MavenStyleRepository repo, List<GroupAndArtifactId> rootGroupAndArtifactIds) {
                this.repo = repo;
                this.rootGroupAndArtifactIds = rootGroupAndArtifactIds;

                Set<Id> unresolved = new HashSet<>();
                rootGroupAndArtifactIds.forEach(
                        rootGroupAndArtifactId -> {
                            var metaData = repo.metaData(rootGroupAndArtifactId).orElseThrow();
                            var metaDataId = metaData.latestMetaDataId().orElseThrow();
                            var optionalPom = repo.pom(rootGroupAndArtifactId);

                            if (optionalPom.isPresent() && optionalPom.get() instanceof Pom pom) {
                                pom.requiredDependencies()
                                        .filter(dependencyId -> !dependencyId.scope.equals(Scope.TEST))
                                        .forEach(
                                                dependencyId -> {
                                                    add(metaDataId, dependencyId);
                                                    unresolved.add(dependencyId);
                                                });
                            }
                        });

                while (!unresolved.isEmpty()) {
                    var resolveSet = new HashSet<>(unresolved);
                    unresolved.clear();
                    resolveSet.forEach(id -> {
                                if (repo.pom(id) instanceof Optional<Pom> p && p.isPresent()) {
                                    p.get()
                                            .requiredDependencies()
                                            .filter(dependencyId -> !dependencyId.scope.equals(Scope.TEST))
                                            .forEach(
                                                    dependencyId -> {
                                                        unresolved.add(dependencyId);
                                                        add(id, dependencyId);
                                                    });
                                }
                            });
                }
                removeUNSPECIFIED();
            }

            @Override
            public List<ClassPathEntry> classPathEntries() {
                return classPath().classPathEntries();
            }

            ClassPath classPath() {

                ClassPath jars = ClassPath.of();
                nodes
                        .keySet()
                        .forEach(
                                id -> {
                                    Optional<Pom> optionalPom = repo.pom(id);
                                    if (optionalPom.isPresent() && optionalPom.get() instanceof Pom pom) {
                                        jars.add(pom.getJar());
                                    } else {
                                        throw new RuntimeException("No pom for " + id + " needed by " + id);
                                    }
                                });
                return jars;
            }
        }

        public ClassPathEntryProvider classPathEntries(String... rootGroupAndArtifactIds) {
            return classPathEntries(Stream.of(rootGroupAndArtifactIds).map(GroupAndArtifactId::of).toList());
        }

        public ClassPathEntryProvider classPathEntries(GroupAndArtifactId... rootGroupAndArtifactIds) {
            return classPathEntries(List.of(rootGroupAndArtifactIds));
        }

        public ClassPathEntryProvider classPathEntries(List<GroupAndArtifactId> rootGroupAndArtifactIds) {
          StringBuilder sb = new StringBuilder();
          rootGroupAndArtifactIds.forEach(groupAndArtifactId->sb.append(sb.isEmpty() ?"":"-").append(groupAndArtifactId.groupId+"-"+groupAndArtifactId.artifactId));
          System.out.println(sb);
          ClassPathEntryProvider classPathEntries=null;
          var pathFileName = sb+"-path.xml";
          var pathFile = dir.xmlFile(pathFileName);
          if (pathFile.exists()){
              System.out.println(pathFileName + " exists " + pathFile.path().toString());
              XMLNode path = new XMLNode(pathFile.path());
              ClassPath classPath = ClassPath.of();
              path.nodes(path.xpath("/path/jar/text()")).forEach(e->
                      classPath.add(dir.jarFile(e.getNodeValue()))
              );
              classPathEntries = classPath;
          }else {
             var finalClassPathEntries =  new Dag(this, rootGroupAndArtifactIds);
                  XMLNode.create("path", xml-> {
                      finalClassPathEntries.classPathEntries().forEach(cpe ->
                              xml.element("jar",jar->jar.text(dir.path().relativize(cpe.path()).toString()))
                      );
                  }).write(pathFile);
             System.out.println("created "+pathFile.path());
             classPathEntries = finalClassPathEntries;
          }
            return classPathEntries;
        }

        public record VersionId(Integer maj, Integer min, Integer point, String classifier)
                implements Comparable<VersionId> {
            static Integer integerOrNull(String s) {
                return (s == null || s.isEmpty()) ? null : Integer.parseInt(s);
            }

            public static Pattern pattern = Pattern.compile("^(\\d+)(?:\\.(\\d+)(?:\\.(\\d+)(.*))?)?$");
            static VersionId UNSPECIFIED = new VersionId(null, null, null, null);

            static VersionId of(String version) {
                Matcher matcher = pattern.matcher(version);
                if (matcher.matches()) {
                    return new VersionId(
                            integerOrNull(matcher.group(1)),
                            integerOrNull(matcher.group(2)),
                            integerOrNull(matcher.group(3)),
                            matcher.group(4));
                } else {
                    return UNSPECIFIED;
                }
            }

            int cmp(Integer v1, Integer v2) {
                if (v1 == null && v2 == null) {
                    return 0;
                }
                if (v1 == null) {
                    return -v2;
                } else if (v2 == null) {
                    return v1;
                } else {
                    return v1 - v2;
                }
            }

            @Override
            public int compareTo(VersionId o) {
                if (cmp(maj(), o.maj()) == 0) {
                    if (cmp(min(), o.min()) == 0) {
                        if (cmp(point(), o.point()) == 0) {
                            return classifier().compareTo(o.classifier());
                        } else {
                            return cmp(point(), o.point());
                        }
                    } else {
                        return cmp(min(), o.min());
                    }
                } else {
                    return cmp(maj(), o.maj());
                }
            }

            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                if (maj() != null) {
                    sb.append(maj());
                    if (min() != null) {
                        sb.append(".").append(min());
                        if (point() != null) {
                            sb.append(".").append(point());
                            if (classifier() != null) {
                                sb.append(classifier());
                            }
                        }
                    }
                } else {
                    sb.append("UNSPECIFIED");
                }
                return sb.toString();
            }
        }

        public record GroupId(String string) {
            public static GroupId of(String s) {
                return new GroupId(s);
            }

            @Override
            public String toString() {
                return string;
            }
        }

        public record ArtifactId(String string) {
            static ArtifactId of(String string) {
                return new ArtifactId(string);
            }

            @Override
            public String toString() {
                return string;
            }
        }

        public record MetaDataId(
                MavenStyleRepository mavenStyleRepository,
                GroupAndArtifactId groupAndArtifactId,
                VersionId versionId,
                Set<String> downloadables,
                Set<String> tags)
                implements Id {

            public URL pomURL() {
                return url("pom");
            }

            public URL jarURL() {
                return url("jar");
            }

            public XMLNode getPom() {
                if (downloadables.contains(".pom")) {
                    return mavenStyleRepository.queryAndCache(
                            url("pom"), mavenStyleRepository.dir.xmlFile(artifactAndVersion() + ".pom"));
                } else {
                    throw new IllegalStateException("no pom");
                }
            }

            @Override
            public String toString() {
                return groupAndArtifactId().toString() + "." + versionId();
            }
        }

        public MavenStyleRepository(RepoDir dir) {
            this.dir = dir.create();
        }

        JarFile queryAndCache(URL query, JarFile jarFile) {
            try {
                if (!jarFile.exists()) {
                    print("Querying and caching " + jarFile.fileName());
                    println(" downloading " + query);
                    curl(query, jarFile.path());
                } else {
                    // println("Using cached " + jarFile.fileName());

                }
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
            return jarFile;
        }

        XMLNode queryAndCache(URL query, XMLFile xmlFile) {
            XMLNode xmlNode = null;
            try {
                if (!xmlFile.exists()) {
                    print("Querying and caching " + xmlFile.fileName());
                    println(" downloading " + query);
                    xmlNode = new XMLNode(query);
                    xmlNode.write(xmlFile.path().toFile());
                } else {
                    // println("Using cached " + xmlFile.fileName());
                    xmlNode = new XMLNode(xmlFile.path());
                }
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
            return xmlNode;
        }

        public record MetaData(
                MavenStyleRepository mavenStyleRepository,
                GroupAndArtifactId groupAndArtifactId,
                XMLNode xmlNode) {

            public Stream<MetaDataId> metaDataIds() {
                return xmlNode
                        .xmlNodes(xmlNode.xpath("/response/result/doc"))
                        .map(
                                xmln ->
                                        new MetaDataId(
                                                this.mavenStyleRepository,
                                                bldr.Bldr.MavenStyleRepository.GroupAndArtifactId.of(
                                                        bldr.Bldr.MavenStyleRepository.GroupId.of(xmln.xpathQueryString("str[@name='g']/text()")),
                                                        bldr.Bldr.MavenStyleRepository.ArtifactId.of(xmln.xpathQueryString("str[@name='a']/text()"))),
                                                bldr.Bldr.MavenStyleRepository.VersionId.of(xmln.xpathQueryString("str[@name='v']/text()")),
                                                new HashSet<>(
                                                        xmln.nodes(xmln.xpath("arr[@name='ec']/str/text()"))
                                                                .map(Node::getNodeValue)
                                                                .toList()),
                                                new HashSet<>(
                                                        xmln.nodes(xmln.xpath("arr[@name='tags']/str/text()"))
                                                                .map(Node::getNodeValue)
                                                                .toList())));
            }

            public Stream<MetaDataId> sortedMetaDataIds() {
                return metaDataIds().sorted(Comparator.comparing(MetaDataId::versionId));
            }

            public Optional<MetaDataId> latestMetaDataId() {
                return metaDataIds().max(Comparator.comparing(MetaDataId::versionId));
            }

            public Optional<MetaDataId> getMetaDataId(VersionId versionId) {
                return metaDataIds().filter(id -> versionId.compareTo(id.versionId()) == 0).findFirst();
            }
        }

        public Optional<MetaData> metaData(String groupId, String artifactId) {
            return metaData(GroupAndArtifactId.of(groupId, artifactId));
        }

        public Optional<MetaData> metaData(GroupAndArtifactId groupAndArtifactId) {
            try {
                var query = "g:" + groupAndArtifactId.groupId() + " AND a:" + groupAndArtifactId.artifactId();
                URL rowQueryUrl =
                        new URI(
                                searchBase
                                        + "select?q="
                                        + URLEncoder.encode(query, StandardCharsets.UTF_8)
                                        + "&core=gav&wt=xml&rows=0")
                                .toURL();
                var rowQueryResponse = new XMLNode(rowQueryUrl);
                var numFound = rowQueryResponse.xpathQueryString("/response/result/@numFound");

                URL url =
                        new URI(
                                searchBase
                                        + "select?q="
                                        + URLEncoder.encode(query, StandardCharsets.UTF_8)
                                        + "&core=gav&wt=xml&rows="
                                        + numFound)
                                .toURL();
                try {
                    // println(url);
                    var xmlNode =
                            queryAndCache(url, dir.xmlFile(groupAndArtifactId.artifactId() + ".meta.xml"));
                    // var numFound2 = xmlNode.xpathQueryString("/response/result/@numFound");
                    // var start = xmlNode.xpathQueryString("/response/result/@start");
                    // var rows =
                    // xmlNode.xpathQueryString("/response/lst[@name='responseHeader']/lst[@name='params']/str[@name='rows']/text()");
                    // println("numFound = "+numFound+" rows ="+rows+ " start ="+start);
                    if (numFound.isEmpty() || numFound.equals("0")) {
                        return Optional.empty();
                    } else {
                        return Optional.of(new MetaData(this, groupAndArtifactId, xmlNode));
                    }
                } catch (Throwable e) {
                    throw new RuntimeException(e);
                }
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class IntelliJ {
        public static class IntellijArtifact {
            Dir projectDir;
            XMLNode root;

            Stream<XMLNode> query(String xpath) {
                return root.nodes(root.xpath(xpath)).map(e -> new XMLNode((Element) e));
            }

            IntellijArtifact(Dir projectDir, XMLNode root) {
                this.projectDir = projectDir;
                this.root = root;
            }
        }

        public static class Workspace extends IntellijArtifact {

            record Application(XMLNode xmlNode) {
            }

            List<Application> applications;

            Workspace(Dir projectDir, XMLNode root) {
                super(projectDir, root);
                this.applications =
                        query("/project/component[@name='RunManager']/configuration")
                                .map(Application::new)
                                .toList();
            }
        }

        public static class Compiler extends IntellijArtifact {
            public record JavacSettings(XMLNode xmlNode) {
                public String getAdditionalOptions() {
                    return xmlNode.xpathQueryString("option[@name='ADDITIONAL_OPTIONS_STRING']/@value");
                }
            }

            public JavacSettings javacSettings;

            Compiler(Dir projectDir, XMLNode root) {
                super(projectDir, root);
                this.javacSettings =
                        new JavacSettings(query("/project/component[@name='JavacSettings']").findFirst().get());
            }
        }

        public static class ImlGraph extends IntellijArtifact {
            public record Module(Path imlPath, XMLNode xmlNode) {
                @Override
                public String toString() {
                    return name();
                }

                public String name() {
                    return imlPath.getFileName().toString();
                }

                public SourcePath getSourcePath() {
                    return null;
                }

                Stream<XMLNode> query(String xpath) {
                    return xmlNode.nodes(xmlNode.xpath(xpath)).map(e -> new XMLNode((Element) e));
                }
            }

            Stream<XMLNode> query(String xpath) {
                return root.nodes(root.xpath(xpath)).map(e -> new XMLNode((Element) e));
            }

            Set<Module> modules = new HashSet<>();
            public Map<Module, List<Module>> fromToDependencies = new HashMap<>();
            Map<Module, List<Module>> toFromDependencies = new HashMap<>();

            ImlGraph(Dir projectDir, XMLNode root) {
                super(projectDir, root);
                Map<String, Module> nameToModule = new HashMap<>();
                query("/project/component[@name='ProjectModuleManager']/modules/module")
                        .map(
                                xmlNode ->
                                        Path.of(
                                                xmlNode
                                                        .attrMap
                                                        .get("filepath")
                                                        .replace("$PROJECT_DIR$", projectDir.path().toString())))
                        .map(path -> new Module(path, new XMLNode(path)))
                        .forEach(
                                module -> {
                                    modules.add(module);
                                    nameToModule.put(module.name(), module);
                                });
                modules.forEach(
                        module ->
                                module
                                        .xmlNode
                                        .nodes(root.xpath("/module/component/orderEntry[@type='module']"))
                                        .map(e -> new XMLNode((Element) e))
                                        .forEach(
                                                e -> {
                                                    var dep = nameToModule.get(e.attrMap.get("module-name") + ".iml");
                                                    fromToDependencies.computeIfAbsent(module, _ -> new ArrayList<>()).add(dep);
                                                    toFromDependencies.computeIfAbsent(dep, _ -> new ArrayList<>()).add(module);
                                                }));
            }
        }

        public static class Project {
            public Dir intellijDir;
            public ImlGraph imlGraph;
            public Workspace workSpace;
            public Compiler compiler;

            public Project(Dir intellijDir) {
                this.intellijDir = intellijDir;
                var ideaDir = intellijDir.existingDir(".idea");
                imlGraph = new ImlGraph(intellijDir, new XMLNode(ideaDir.xmlFile("modules.xml").path()));
                workSpace = new Workspace(intellijDir, new XMLNode(ideaDir.xmlFile("workspace.xml").path()));
                compiler = new Compiler(intellijDir, new XMLNode(ideaDir.xmlFile("compiler.xml").path()));
            }
        }

    }
}

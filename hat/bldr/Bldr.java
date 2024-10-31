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
import java.nio.file.FileVisitOption;
import java.nio.file.FileVisitResult;
import java.nio.file.FileVisitor;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.PosixFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
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
    public interface PathHolder  {
        Path path();
        default String name(){
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

    }

    public interface TargetDirProvider extends PathHolder {
        Path targetDir();
    }

    public interface JavaSourceDirProvider {
        Path javaSourceDir();
    }

    public interface ResourceDirProvider {
        DirPathHolder resourcesDir();
    }

    public interface  DirPathHolder<T extends DirPathHolder<T>> extends PathHolder {
         default Path path(String subdir){
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
            return find( Files::isRegularFile);
        }

        default Stream<Path> findDirs() {
            return find( Files::isDirectory);
        }

        default Stream<Path> findFiles(Predicate<Path> predicate) {
            return findFiles().filter(predicate);
        }

        default Stream<SearchableTextFile> findTextFiles(String... suffixes) {
            return findFiles().map(SearchableTextFile::new).filter(searchableTextFile -> searchableTextFile.hasSuffix(suffixes));
        }

        default Stream<Path> findDirs( Predicate<Path> predicate) {
            return find( Files::isDirectory).filter(predicate);
        }


        default boolean exists(){
             return Files.exists(path()) && Files.isDirectory(path());
        }


    }

    public interface FilePathHolder extends PathHolder { }

    public interface ClassPathEntry extends PathHolder { }
    public record CMakeBuildDir(Path path) implements  BuildDirHolder<CMakeBuildDir> {
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
    public interface BuildDirHolder<T extends BuildDirHolder<T>> extends DirPathHolder<T> {
        T create();
        T remove();
        default void clean(){
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
                    Files.walk(path).sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
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

        public static ClassDir temp(String javacclasses) {
            try {
                return of(Files.createTempDirectory("javacClasses"));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public ClassDir create(){
            return ClassDir.of(mkdir(path()));
        }
        @Override
        public ClassDir remove(){
            return ClassDir.of(rmdir(path()));
        }
    }
    public record Dir(Path path) implements  DirPathHolder<Dir> {
        public static Dir of(Path path){
           return new Dir(path);
       }
        public static Dir of(String string){
            return of (Path.of(string));
        }
        public static Dir current(){
            return of(Path.of(System.getProperty("user.dir")));
        }
        public Dir parent(){
            return of(path().getParent());
        }

        public  Dir dir(String subdir){
            return Dir.of(path(subdir));
        }
        public Stream<Dir> forEachSubDirectory(String ... dirNames){
            return Stream.of(dirNames).map(dirName->path().resolve(dirName)).filter(Files::isDirectory).map(Dir::new);
        }

    }
    public record RootDirAndSubPath(DirPathHolder<?> root, Path path) {
        Path relativize() {
            return root().path().relativize(path());
        }
    }
    public record BuildDir(Path path) implements ClassPathEntry, BuildDirHolder<BuildDir> {
        public static BuildDir of(Path path){
            return new BuildDir(path);
        }

        public JarFile jarFile(String name) {
            return JarFile.of(path().resolve(name));
        }
        public CMakeBuildDir cMakeBuildDir(String name) {
            return CMakeBuildDir.of(path().resolve(name));
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

        public BuildDir dir(String subdir){
            return BuildDir.of(path(subdir));
        }

    }

    public record JarFile(Path path) implements ClassPathEntry, FilePathHolder {
        public static JarFile of(Path path) {
            return new JarFile(path);
        }
    }

    public record SourcePathEntry(Path path) implements DirPathHolder<SourcePathEntry> {
    }

    public interface TextFile extends FilePathHolder{

    }

    public interface SourceFile extends TextFile {
    }

    public static class JavaSourceFile extends SimpleJavaFileObject implements SourceFile {
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

    public record CppSourceFile(Path path) implements SourceFile {
    }

    public record CppHeaderSourceFile(Path path) implements SourceFile {
    }


    public record ClassPath(List<ClassPathEntry> entries) {
    }

    public record SourcePath(List<SourcePathEntry> entries) {
    }

    public record XMLFile(Path path) implements TextFile {
    }

    public interface OS {
        String arch();

        String name();

        String version();

        static final String MacName = "Mac OS X";
        static final String LinuxName = "Linux";


        record Linux(String arch, String name, String version) implements OS {
        }

        record Mac(String arch, String name, String version) implements OS {
            public Path appLibFrameworks() {
                return Path.of("/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
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

    public record Java(String version, File home) {
    }

    public static Java java = new Java(System.getProperty("java.version"), new File(System.getProperty("java.home")));

    public record User(File home, File pwd) {
    }

    public static User user = new User(new File(System.getProperty("user.home")), new File(System.getProperty("user.dir")));


    /*
        static class POM {
            static Pattern varPattern = Pattern.compile("\\$\\{([^}]*)\\}");
            static public String varExpand(Map<String, String> props, String value) { // recurse
                String result = value;
                if (varPattern.matcher(value) instanceof Matcher matcher && matcher.find()) {
                    var v = matcher.groupId(1);
                    result = varExpand(props, value.substring(0, matcher.start())
                            + (v.startsWith("env")
                            ? System.getenv(v.substring(4))
                            : props.get(v))
                            + value.substring(matcher.end()));
                    //out.println("incomming ='"+value+"'  v= '"+v+"' value='"+value+"'->'"+result+"'");
                }
                return result;
            }

            POM(Path dir) throws Throwable {
                var topPom = new XMLNode(new File(dir.toFile(), "pom.xml"));
                var babylonDirKey = "babylon.dir";
                var spirvDirKey = "beehive.spirv.toolkit.dir";
                var hatDirKey = "hat.dir";
                var interestingKeys = Set.of(spirvDirKey, babylonDirKey, hatDirKey);
                var requiredDirKeys = Set.of(babylonDirKey, hatDirKey);
                var dirKeyToDirMap = new HashMap<String, File>();
                var props = new HashMap<String, String>();

                topPom.children.stream().filter(e -> e.element.getNodeName().equals("properties")).forEach(properties ->
                        properties.children.stream().forEach(property -> {
                            var key = property.element.getNodeName();
                            var value = varExpand(props, property.element.getTextContent());
                            props.put(key, value);
                            if (interestingKeys.contains(key)) {
                                var file = new File(value);
                                if (requiredDirKeys.contains(key) && !file.exists()) {
                                    System.err.println("ERR pom.xml has property '" + key + "' with value '" + value + "' but that dir does not exists!");
                                    System.exit(1);
                                }
                                dirKeyToDirMap.put(key, file);
                            }
                        })
                );
                for (var key : requiredDirKeys) {
                    if (!props.containsKey(key)) {
                        System.err.println("ERR pom.xml expected to have property '" + key + "' ");
                        System.exit(1);
                    }
                }
            }
        }
    */

    public static String charSeparatedClassPath(List<ClassPathEntry> classPathEntries) {
        StringBuilder sb = new StringBuilder();
        classPathEntries.forEach(classPathEntry -> {
            if (!sb.isEmpty()) {
                sb.append(File.pathSeparatorChar);
            }
            sb.append(classPathEntry.path());
        });
        return sb.toString();
    }
    public static String charSeparatedDirPathHolders(List<DirPathHolder<?>> dirPathHolderEntries) {
        StringBuilder sb = new StringBuilder();
        dirPathHolderEntries.forEach(dirPathHolderEntry -> {
            if (!sb.isEmpty()) {
                sb.append(File.pathSeparatorChar);
            }
            sb.append(dirPathHolderEntry.path());
        });
        return sb.toString();
    }

    public abstract static class Builder<T extends Builder<T>> {
        @SuppressWarnings("unchecked") T self() {
            return (T) this;
        }

        public List<String> opts = new ArrayList<>();
        public boolean verbose;
        public T verbose(boolean verbose) {
            this.verbose= verbose;
            return self();
        }
        public T verbose() {
            verbose(true);
            return self();
        }

        public abstract T show(Consumer<String> stringConsumer);

        public T opts(List<String> opts) {
            this.opts.addAll(opts);
            return self();
        }

        public T opts(String... opts) {
            opts(Arrays.asList(opts));
            return self();
        }

        public T basedOn(T stem) {
            if (stem != null) {
                opts.addAll(stem.opts);
            }
            return self();
        }

        public T when(boolean condition, Consumer<T> consumer) {
            if (condition) {
                consumer.accept(self());
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

    public static abstract class ExecBuilder<T extends ExecBuilder<T>> extends Builder<T> {
        abstract public List<String> execOpts();

        public void execInheritIO(Path path) {
            try {
                var processBuilder = new ProcessBuilder();

                if (path != null) {
                    processBuilder.directory(path.toFile());
                }
                processBuilder
                        .inheritIO()
                        .command(execOpts());
                var process = processBuilder
                        .start();
                if (verbose){
                   print(execOpts());
                    // show((s)->print(execOpts()));
                }
                process.waitFor();

            } catch (InterruptedException ie) {
                System.out.println(ie);
            } catch (IOException ioe) {
                System.out.println(ioe);
            }
        }

        public void execInheritIO() {
            execInheritIO(null);
        }
    }

    public static class JavacBuilder extends Builder<JavacBuilder> {
        public ClassDir classDir;
        public List<DirPathHolder<?>> sourcePath ;
        public List<ClassPathEntry> classPath;

        @Override
        public JavacBuilder show(Consumer<String> stringConsumer) {
             return self();
        }

        public JavacBuilder basedOn(JavacBuilder stem) {
            super.basedOn(stem);
            if (stem != null) {
                if (stem.classDir != null) {
                    this.classDir = stem.classDir;
                }
                if (stem.sourcePath != null) {
                    this.sourcePath = new ArrayList<>(stem.sourcePath);
                }
                if (stem.classPath != null) {
                    this.classPath = new ArrayList<>(stem.classPath);
                }
            }
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

        public JavacBuilder source_path(List<DirPathHolder<?>> sourcePaths) {
            this.sourcePath = this.sourcePath == null ? new ArrayList<>() : this.sourcePath;
            this.sourcePath.addAll(sourcePaths);
            return this;
        }

        public JavacBuilder source_path(DirPathHolder<?>... sourcePaths) {
            return source_path(List.of(sourcePaths));
        }

        public JavacBuilder class_path(ClassPathEntry... classPathEntries) {
            this.classPath = this.classPath == null ? new ArrayList<>() : this.classPath;
            this.classPath.addAll(Arrays.asList(classPathEntries));
            return this;
        }

    }





    public static JavacBuilder javac(JavacBuilder javacBuilder) {
        try {
            if (javacBuilder.classDir == null) {
                javacBuilder.classDir = ClassDir.temp("javacclasses");
              }
            javacBuilder.opts.addAll(List.of("-d", javacBuilder.classDir.path().toString()));
            javacBuilder.classDir.clean();


            if (javacBuilder.classPath != null) {
                javacBuilder.opts.addAll(List.of("--class-path", charSeparatedClassPath(javacBuilder.classPath)));
            }

            javacBuilder.opts.addAll(List.of("--source-path", charSeparatedDirPathHolders(javacBuilder.sourcePath)));
            var compilationUnits = new ArrayList<JavaSourceFile>();
            javacBuilder.sourcePath.forEach(entry ->
                    entry.findFiles( file -> file.toString().endsWith(".java"))
                            .map(JavaSourceFile::new)
                            .forEach(compilationUnits::add));

            DiagnosticListener<JavaFileObject> dl = (diagnostic) -> {
                if (!diagnostic.getKind().equals(Diagnostic.Kind.NOTE)) {
                    System.out.println(diagnostic.getKind()
                            + " " + diagnostic.getLineNumber() + ":" + diagnostic.getColumnNumber() + " " + diagnostic.getMessage(null));
                }
            };

            //   List<RootAndPath> pathsToJar = new ArrayList<>();
            JavaCompiler javac = javax.tools.ToolProvider.getSystemJavaCompiler();
            JavaCompiler.CompilationTask compilationTask = (javac.getTask(
                    new PrintWriter(System.err),
                    javac.getStandardFileManager(dl, null, null),
                    dl,
                    javacBuilder.opts,
                    null,
                    compilationUnits

            ));
            ((com.sun.source.util.JavacTask) compilationTask)
                    .generate();
            //.forEach(fileObject -> pathsToJar.add(new RootAndPath(javacBuilder.classesDir, Path.of(fileObject.toUri()))));


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

    public static class JarBuilder extends Builder<JarBuilder> {
        public JarFile jar;
        public JavacBuilder javacBuilder;
        public List<DirPathHolder<?>> dirList;

        public JarBuilder basedOn(JarBuilder stem) {
            super.basedOn(stem);
            if (stem != null) {
                if (stem.jar != null) {
                    this.jar = stem.jar;
                }
                if (stem.dirList != null) {
                    this.dirList = new ArrayList<>(stem.dirList);
                }
            }
            return this;
        }

        public JarBuilder jar(JarFile jar) {
            this.jar = jar;
            return this;
        }
        public JarBuilder javac( JavacBuilder javacBuilder) {
            this.javacBuilder = Bldr.javac(javacBuilder);
            this.dirList = (this.dirList == null) ? new ArrayList<>() : this.dirList;
            this.dirList.add(this.javacBuilder.classDir);
            return this;
        }
        public JarBuilder javac(Consumer<JavacBuilder> javacBuilderConsumer) {
            this.javacBuilder = new JavacBuilder();
            javacBuilderConsumer.accept(this.javacBuilder);
            return javac(this.javacBuilder);
        }
        public JarBuilder dir_list(Predicate<DirPathHolder<?>> predicate, DirPathHolder<?>... dirs) {
            Stream.of(dirs).filter(predicate).forEach(optionalDir->{
                this.dirList = (this.dirList == null) ? new ArrayList<>() : this.dirList;
                this.dirList.add(optionalDir);
            });
            return this;
        }
        public JarBuilder dir_list(DirPathHolder<?>... dirs) {
            return dir_list(_->true, dirs);
        }
        @Override
        public JarBuilder show(Consumer<String> stringConsumer) {
            return self();
        }
    }

    public static JarFile jar(Consumer<JarBuilder> jarBuilderConsumer) {
        try {
            JarBuilder jarBuilder = new JarBuilder();
            jarBuilderConsumer.accept(jarBuilder);

            List<RootDirAndSubPath> pathsToJar = new ArrayList<>();
            var jarStream = new JarOutputStream(Files.newOutputStream(jarBuilder.jar.path()));
            jarBuilder.dirList.forEach(root -> root
                    .findFiles()
                    .map(path -> new RootDirAndSubPath(root, path))
                    .forEach(pathsToJar::add));
            pathsToJar.stream().sorted(Comparator.comparing(RootDirAndSubPath::path)).forEach(rootAndPath -> {
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

    public static class JavaBuilder extends ExecBuilder<JavaBuilder> {
        public Path jdk = Path.of(System.getProperty("java.home"));
        public String mainClass;
        public List<ClassPathEntry> classPath;
        public List<DirPathHolder<?>> libraryPath;
        public List<String> vmopts = new ArrayList<>();
        public List<String> args = new ArrayList<>();
        @Override
        public JavaBuilder show(Consumer<String> stringConsumer) {
            return self();
        }
        public JavaBuilder vmopts(List<String> opts) {
            this.vmopts.addAll(opts);
            return self();
        }

        public JavaBuilder vmopts(String... opts) {
            vmopts(Arrays.asList(opts));
            return self();
        }


        public JavaBuilder args(List<String> opts) {
            this.args.addAll(opts);
            return self();
        }

        public JavaBuilder args(String... opts) {
            args(Arrays.asList(opts));
            return self();
        }


        public JavaBuilder basedOn(JavaBuilder stem) {
            super.basedOn(stem);
            if (stem != null) {
                vmopts.addAll(stem.vmopts);
                args.addAll(stem.args);
                if (stem.mainClass != null) {
                    this.mainClass = stem.mainClass;
                }
                if (stem.jdk != null) {
                    this.jdk = stem.jdk;
                }
                if (stem.classPath != null) {
                    this.classPath = new ArrayList<>(stem.classPath);
                }

                opts.addAll(stem.opts);

            }
            return this;
        }

        public JavaBuilder main_class(String mainClass) {
            this.mainClass = mainClass;
            return this;
        }

        public JavaBuilder jdk(Path jdk) {
            this.jdk = jdk;
            return this;
        }

        public JavaBuilder class_path(List<ClassPathEntry> classPathEntries) {
            this.classPath = (this.classPath == null) ? new ArrayList<>() : this.classPath;
            this.classPath.addAll(classPathEntries);
            return this;
        }

        public JavaBuilder class_path(ClassPathEntry... classPathEntries) {
            return this.class_path(List.of(classPathEntries));
        }
        public JavaBuilder library_path(List<DirPathHolder<?>> libraryPathEntries) {
            this.libraryPath = (this.libraryPath == null) ? new ArrayList<>() : this.libraryPath;
            this.libraryPath.addAll(libraryPathEntries);
            return this;
        }
        public JavaBuilder library_path(DirPathHolder<?>... libraryPathEntries) {
            return this.library_path(List.of(libraryPathEntries));
        }

        @Override
        public List<String> execOpts() {
            List<String> execOpts = new ArrayList<>();
            execOpts.add(jdk.resolve("bin/java").toString());
            execOpts.addAll(vmopts);
            if (classPath != null) {
                execOpts.addAll(List.of("--class-path", charSeparatedClassPath(classPath)));
            }
            if (libraryPath!= null) {
                execOpts.add("-Djava.library.path="+ charSeparatedDirPathHolders(libraryPath));
            }
            execOpts.add(mainClass);
            execOpts.addAll(args);
            return execOpts;
        }
    }
    public static JavaBuilder java(JavaBuilder javaBuilder) {
        javaBuilder.execInheritIO();
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


    public static class CMakeBuilder extends ExecBuilder<CMakeBuilder> {
        public List<String> libraries = new ArrayList<>();
        public CMakeBuildDir cmakeBuildDir;
        public Dir sourceDir;
        private Path output;
        public BuildDir copyToDir;

        public CMakeBuilder() {
            opts.add("cmake");
        }
        @Override
        public CMakeBuilder show(Consumer<String> stringConsumer) {
            return self();
        }
        public CMakeBuilder basedOn(CMakeBuilder stem) {
            // super.basedOn(stem); you will get two cmakes ;)
            if (stem != null) {
                if (stem.output != null) {
                    this.output = stem.output;
                }
                if (stem.copyToDir != null) {
                    this.copyToDir = stem.copyToDir;
                }
                if (stem.libraries != null) {
                    this.libraries = new ArrayList<>(stem.libraries);
                }
                if (stem.cmakeBuildDir != null) {
                    this.cmakeBuildDir = stem.cmakeBuildDir;
                }
                if (stem.sourceDir != null) {
                    this.sourceDir = stem.sourceDir;
                }
            }
            return this;
        }

        public CMakeBuilder build_dir(CMakeBuildDir cmakeBuildDir) {
            this.cmakeBuildDir = cmakeBuildDir;
            opts("-B", cmakeBuildDir.path.toString());
            return this;
        }
        public CMakeBuilder copy_to(BuildDir copyToDir) {
            this.copyToDir = copyToDir;
            opts("-DHAT_TARGET=" +this.copyToDir.path().toString());
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

        @Override
        public List<String> execOpts() {
            return opts;
        }
    }

    public static void cmake(Consumer<CMakeBuilder> cmakeBuilderConsumer) {

        CMakeBuilder cmakeBuilder = new CMakeBuilder();
        cmakeBuilderConsumer.accept(cmakeBuilder);
        cmakeBuilder.cmakeBuildDir.create();
        cmakeBuilder.execInheritIO();
    }


    static Path unzip(Path in, Path dir) {
        try {
            Files.createDirectories(dir);
            ZipFile zip = new ZipFile(in.toFile());
            zip.entries().asIterator().forEachRemaining(entry -> {
                try {
                    String currentEntry = entry.getName();

                    Path destFile = dir.resolve(currentEntry);
                    //destFile = new File(newPath, destFile.getName());
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

    public static class JExtractBuilder extends ExecBuilder<JExtractBuilder> {
        public List<String> compileFlags = new ArrayList<>();
        public List<Path> libraries = new ArrayList<>();
        public List<Path> headers = new ArrayList<>();
        public Path cwd;

        public Path home;
        private String targetPackage;
        private Path output;
        @Override
        public JExtractBuilder show(Consumer<String> stringConsumer) {
            return self();
        }
        public JExtractBuilder() {
            opts.add("jextract");
        }

        public JExtractBuilder basedOn(JExtractBuilder stem) {
            super.basedOn(stem);
            if (stem != null) {
                if (stem.output != null) {
                    this.output = stem.output;
                }
                if (stem.compileFlags != null) {
                    this.compileFlags = new ArrayList<>(stem.compileFlags);
                }
                if (stem.libraries != null) {
                    this.libraries = new ArrayList<>(stem.libraries);
                }
                if (stem.home != null) {
                    this.home = stem.home;
                }
                if (stem.cwd != null) {
                    this.cwd = stem.cwd;
                }
                if (stem.headers != null) {
                    this.headers = new ArrayList<>(stem.headers);
                }
            }
            return this;
        }


        public JExtractBuilder cwd(Path cwd) {
            this.cwd = cwd;
            return this;
        }

        public JExtractBuilder home(Path home) {
            this.home = home;
            opts.set(0, home.resolve("bin/jextract").toString());
            return this;
        }

        public JExtractBuilder opts(String... opts) {
            this.opts.addAll(Arrays.asList(opts));
            return this;
        }

        public JExtractBuilder target_package(String targetPackage) {
            this.targetPackage = targetPackage;
            opts("--target-package", targetPackage);
            return this;
        }

        public JExtractBuilder output(Path output) {
            this.output = output;
            opts("--output", output.toString());
            return this;
        }

        public JExtractBuilder library(Path... libraries) {
            this.libraries.addAll(Arrays.asList(libraries));
            for (Path library : libraries) {
                opts("--library", ":" + library);
            }
            return this;
        }

        public JExtractBuilder compile_flag(String... compileFlags) {
            this.compileFlags.addAll(Arrays.asList(compileFlags));
            return this;
        }

        public JExtractBuilder header(Path header) {
            this.headers.add(header);
            this.opts.add(header.toString());
            return this;
        }

        @Override
        public List<String> execOpts() {
            return opts;
        }
    }

    public static void jextract(Consumer<JExtractBuilder> jextractBuilderConsumer) {
        JExtractBuilder extractConfig = new JExtractBuilder();
        jextractBuilderConsumer.accept(extractConfig);
        System.out.println(extractConfig.opts);
        var compilerFlags = extractConfig.cwd.resolve("compiler_flags.txt");
        try {
            PrintWriter compilerFlagsWriter = new PrintWriter(Files.newOutputStream(compilerFlags));
            compilerFlagsWriter.println(extractConfig.compileFlags);
            compilerFlagsWriter.close();
            Files.createDirectories(extractConfig.output);
            extractConfig.execInheritIO(extractConfig.cwd);
            Files.deleteIfExists(compilerFlags);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }



    public record SearchableTextFile(Path path) implements TextFile {
        public Stream<Line> lines() {
            try {
                int num[] = new int[]{1};
                return Files.readAllLines(path(), StandardCharsets.UTF_8).stream().map(line -> new Line(line, num[0]++));
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
                .filter(Files::isExecutable).findFirst();
    }

    public static boolean canExecute(String execName) {
        // which and whereis had issues.
        return which(execName).isPresent();
    }

    public static Path untar(Path tarFile, Path dir) {
        try {
            new ProcessBuilder().inheritIO().command("tar", "xvf", tarFile.toString(), "--directory", tarFile.getParent().toString()).start().waitFor();
            return dir;
        } catch (
                InterruptedException e) { // We get IOException if the executable not found, at least on Mac so interuppted means it exists
            return null;
        } catch (IOException e) { // We get IOException if the executable not found, at least on Mac
            //throw new RuntimeException(e);
            return null;
        }
    }




    public record Root(Path path) implements DirPathHolder<Root> {
        public BuildDir buildDir() {
            return BuildDir.of(path("build")).create();
        }

        public BuildDir thirdPartyDir() {
            return BuildDir.of(path("thirdparty")).create();
        }

        public BuildDir repoDir() {
            return BuildDir.of(path("repoDir")).create();
        }

        public Root() {
            this(Path.of(System.getProperty("user.dir")));
        }
    }


        public static Path requireJExtract(Dir thirdParty) {
            var optional = executablesInPath("jextract").findFirst();
            if (optional.isPresent()) {
                println("Found jextract in PATH");
                return optional.get().getParent().getParent(); // we want the 'HOME' dir
            }
            println("No jextract in PATH");
            URL downloadURL = null;
            var extractVersionMaj = "22";
            var extractVersionMin = "5";
            var extractVersionPoint = "33";


            var nameArchTuple = switch (os.name()) {
                case OS.MacName -> "macos";
                default -> os.name().toLowerCase();
            } + '-' + os.arch();

            try {
                downloadURL = new URI("https://download.java.net/java/early_access"
                        + "/jextract/" + extractVersionMaj + "/" + extractVersionMin
                        + "/openjdk-" + extractVersionMaj + "-jextract+" + extractVersionMin + "-" + extractVersionPoint + "_"
                        + nameArchTuple + "_bin.tar.gz").toURL();
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



    public static Stream<Path> executablesInPath(String name) {
        return Arrays.asList(System.getenv("PATH").split(File.pathSeparator)).stream()
                .map(dirName -> Path.of(dirName).resolve(name).normalize())
                .filter(Files::isExecutable);

    }

    public static void sanity(Root hatDir) {
        var rleParserDir = hatDir.path().resolve("examples/life/src/main/java/io");
        Dir.of(hatDir.path).forEachSubDirectory( "hat", "examples", "backends", "docs").forEach(dir ->{
                dir.findFiles()
                        .filter((path)->Pattern.matches("^.*\\.(java|cpp|h|hpp|md)", path.toString()))
                        .forEach(path -> println(path));

                dir.findTextFiles("java", "cpp", "h", "hpp", "md")
                        .forEach(searchableTextFile -> {
                            if (!searchableTextFile.path().getFileName().toString().equals("Makefile") && !searchableTextFile.hasSuffix("md")
                                    && !searchableTextFile.path().startsWith(rleParserDir)
                                    && !searchableTextFile.grep(Pattern.compile("^.*Copyright.*202[4-9].*(Intel|Oracle).*$"))) {
                                System.err.println("ERR MISSING LICENSE " + searchableTextFile.path());
                            }
                            searchableTextFile.lines().forEach(line -> {
                                if (!searchableTextFile.path().getFileName().toString().startsWith("Makefile") && line.grep(Pattern.compile("^.*\\t.*"))) {
                                    System.err.println("ERR TAB " + searchableTextFile.path() + ":" + line.line() + "#" + line.num());
                                }
                                if (line.grep(Pattern.compile("^.* $"))) {
                                    System.err.println("ERR TRAILING WHITESPACE " + searchableTextFile.path() + ":" + line.line() + "#" + line.num());
                                }
                            });
                        });}
        );
    }

    public static <T> T assertOrThrow(T testme, Predicate<T> predicate, String message){
        if (predicate.test(testme)) {
            return testme;
        }else{
            throw new IllegalStateException("FAILED: "+message+" "+testme);
        }
    }

    public static <T extends PathHolder> T assertExists(T testme){
        if (Files.exists(testme.path())) {
            return testme;
        }else{
            throw new IllegalStateException("FAILED: "+testme.path()+" does not exist");
        }
    }
    public static <T extends Path> T assertExists(T path){
        if (Files.exists(path)) {
            return path;
        }else{
            throw new IllegalStateException("FAILED: "+path+" does not exist");
        }
    }

    void main(String[] args) {
        var bldrDir = Dir.current().parent().parent().parent();
        var buildDir =BuildDir.of(bldrDir.path("build")).create();

        jar($->$
                .jar(buildDir.jarFile("bldr.jar"))
                .javac($$->$$
                        .opts(
                                "--source", "24",
                                "--enable-preview",
                                "--add-exports=java.base/jdk.internal=ALL-UNNAMED",
                                "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED"
                        )
                        .class_dir(buildDir.classDir("bld.jar.classes"))
                        .source_path(bldrDir.dir("src/main/java"))
                )
        );
    }
}

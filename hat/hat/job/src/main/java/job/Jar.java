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
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Jar extends DependencyImpl<Jar> implements Dependency.Buildable, Dependency.WithPath, Dependency.ExecutableJar {
    final Set<Path> exclude;

    public void check(){
        if (id.path() != null && !Files.exists(id.path())) {
            System.err.println("The path does not exist: " + id.path());
        }
        if (!Files.exists(javaSourcePath())) {
            var jsp = javaSourcePath();
            System.out.println("Failed to find java source " + jsp + " path for " + id.shortHyphenatedName());
        }
    }

    protected Jar(Project.Id id, Set<Path> exclude, Set<Dependency> dependencies) {
        super(id, dependencies);
        this.exclude = exclude;
        check(); // some targets might skip this
        id.project().add(this);
    }
    public static Jar of(Project.Id id,  Set<Path> exclude, Set<Dependency> dependencies) {
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

    public void forEachEntry(Predicate<JarEntry> jarEntryPredicate,Consumer<JarEntry> jarEntryConsumer) {
       Util.forEachEntry(jarFile(),jarEntryPredicate,jarEntryConsumer);
    }
    public void forEachMatchingEntry(String re, BiConsumer<JarEntry, Matcher> matchingJarEntryConsumer){
        var pattern = Pattern.compile(re);
        forEachEntry(jarEntry->{
            if (pattern.matcher(jarEntry.getName()) instanceof Matcher matched && matched.matches()){
                matchingJarEntryConsumer.accept(jarEntry,matched);
            }
        });
    }
    public void forEachEntry(Consumer<JarEntry> jarEntryConsumer) {
        Util.forEachEntry(jarFile(),jarEntryConsumer);
    }

    private static class JavaSource extends SimpleJavaFileObject {
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
    public boolean clean(boolean verbose) {
        id().project().clean(verbose,null, classesDir(), jarFile());
        return true;
    }

    @Override
    public boolean build() {
        Dag dag = new Dag(dependencies());
        var deps = classPath(dag.ordered());
        StringList javacCombinedOps = StringList.of()
                .add(id().project().javacOpts().vmOpts())
                .add("-d", classesDirName())
                .addIf(!deps.isEmpty(), "--class-path="+deps)
                .add("--source-path=" + javaSourcePathName());

        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();

        id().project().clean(id().project().javacOpts().verbose(), this, classesDir());

        if (Files.exists(javaSourcePath())) {
            try (var files = Files.walk(javaSourcePath())) {
                var listOfSources = files.filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".java") && !exclude.contains(p)).map(JavaSource::new).toList();
                if (id().project().javacOpts().command()){
                    var commandLine = "javac " +
                            javacCombinedOps + " " + String.join(" ", listOfSources.stream().map(JavaSource::getName).collect(Collectors.toList()));
                    System.out.println(commandLine);
                }
                var diagnosticListener = new DiagnosticListener<JavaFileObject>() {
                    @Override
                    public void report(Diagnostic<? extends JavaFileObject> diagnostic) {
                        if (diagnostic.getKind() == Diagnostic.Kind.ERROR) {
                            System.err.println(diagnostic.getKind() + ":"+ diagnostic);
                        } else if (diagnostic.getKind() == Diagnostic.Kind.WARNING) {
                            System.out.println(diagnostic.getKind() + ":"+ diagnostic);
                        } else if (diagnostic.getKind() == Diagnostic.Kind.MANDATORY_WARNING) {
                            System.out.println(diagnostic.getKind() + ":"+ diagnostic);
                        } else if (diagnostic.getKind() == Diagnostic.Kind.NOTE) {
                            if (id().project().javacOpts().verbose()) {
                                System.out.println(diagnostic.getKind() + ":"+ diagnostic);
                            }
                        } else {
                            System.out.println(diagnostic.getKind() + ":"+ diagnostic);
                          //  id().project().reporter.warning(Jar.this, diagnostic.getKind() + ":" + diagnostic.toString());
                        }
                    }
                };
                ((JavacTask) javac.getTask(
                        new PrintWriter(System.err),
                        javac.getStandardFileManager(diagnosticListener, null, null),
                        diagnosticListener,
                        javacCombinedOps.list(),
                        null,
                        listOfSources
                )).generate().forEach(gc -> {
                            if (id().project().javacOpts().verbose()) {
                               System.out.println(gc.getName());
                            }
                        }
                );

                List<Path> dirsToJar = new ArrayList<>(List.of(classesDir()));
                if (Files.exists(javaResourcePath())) {
                    dirsToJar.add(javaResourcePath());
                }

                Manifest manifest = new Manifest();
                Attributes mainAttributes = manifest.getMainAttributes();
                mainAttributes.put(Attributes.Name.MANIFEST_VERSION, "1.0");
                var jarStream = new JarOutputStream(Files.newOutputStream(jarFile()), manifest);
                record RootAndPath(Path root, Path path) {
                }
                if (id().project().javacOpts().command()) {
                    System.out.println( "jar cvf " + jarFile() + " " +
                            String.join(dirsToJar.stream().map(Path::toString).collect(Collectors.joining(" "))));
                }
                if (id().project().javacOpts().progress()) {
                    System.out.println("compiled " + listOfSources.size() + " file" + (listOfSources.size() > 1 ? "s" : "") + " to " + jarFile().getFileName());
                }
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
    public boolean run(JavaConfig javaConfig, Dependency ...unorderedDeps) {
        // unordered deps just contains the min . we need a full ordered dag
        var depsInOrder = Dag.ordered(unorderedDeps);
        StringList stringList = StringList.of()
                .add(Util.currentProcessPath())
                .add(javaConfig.vmOpts())
                .add(
                   "--class-path", classPathWithThisLast(depsInOrder),
                   "-Djava.library.path=" + id().project().buildPath()
                )
                .add(javaConfig.vmOpts())
                .add(javaConfig.mainClassName());
                stringList.add(javaConfig.args());
        if (javaConfig.command()) {
           // id().project().reporter.command(this, stringList.toString());
            System.out.println(stringList.toString());
        }
        var result = ForkExec.forkExec(this, javaConfig.verbose(), id().project().rootPath(), stringList);
        if (result.status() != 0) {
            System.out.println("Java failed to execute, is a valid java in your path ? " + id().fullHyphenatedName());
        }
        return result.status() == 0;
    }

    public  interface JavacConfig extends CommonConfig<JavacConfig> {
        List<String> vmOpts();

        record JavacConfigImpl(boolean command, boolean warnings, boolean progress, boolean verbose, List<String> vmOpts) implements JavacConfig {
        }

        static JavacConfig of(boolean command,boolean warnings, boolean progress, boolean verbose, List<String> vmOpts) {
            return new JavacConfigImpl(command,warnings, progress, verbose, vmOpts);
        }

        interface Builder extends  JavacConfig {
            Builder command(boolean f);
            Builder warnings(boolean f);
            Builder verbose(boolean f);
            Builder progress(boolean f);
            Builder vmOpt(String... s);

            Builder debug();

            Builder enablePreview();

            Builder source(int n);

            Builder addModules(String moduleName);


            class Impl implements Builder {
                boolean command;
                boolean warnings;
                boolean progress;
                boolean verbose;
                List<String> vmOpts = new ArrayList<>();

                @Override
                public Builder command(boolean f) {
                    command = f;
                    return this;
                }
                @Override
                public Builder warnings(boolean f) {
                    warnings = f;
                    return this;
                }

                @Override
                public Builder verbose(boolean f) {
                    verbose = f;
                    return this;
                }
                @Override
                public Builder progress(boolean f) {
                    progress = f;
                    return this;
                }

                @Override
                public Builder vmOpt(String... opts) {
                    vmOpts.addAll(List.of(opts));
                    return this;
                }

                @Override
                public Builder debug() {
                    return vmOpt("-g");
                }

                @Override
                public Builder enablePreview() {
                    return vmOpt("--enable-preview");
                }

                @Override
                public Builder source(int n) {
                    return vmOpt("--source=" + n);
                }

                @Override
                public Builder addModules(String moduleName) {
                    return vmOpt("--add-modules=" + moduleName);
                }

                @Override
                public boolean command() {
                    return command;
                }

                @Override
                public boolean verbose() {
                    return verbose;
                }
                @Override
                public boolean progress() {
                    return progress;
                }
                @Override
                public boolean warnings() {
                    return warnings;
                }
                @Override
                public List<String> vmOpts() {
                   return vmOpts;
                }
            }
        }


        static JavacConfig of(Consumer<Builder> javacOptBuilderConsumer) {
            Builder builder = new Builder.Impl();
            javacOptBuilderConsumer.accept(builder);
            return of(builder.command(),builder.warnings(),builder.progress(), builder.verbose(), builder.vmOpts());
        }

        static JavacConfig of() {
            return of(false, false,false,false, new ArrayList<>());
        }
    }

    public  interface JavaConfig extends CommonConfig<JavaConfig> {

        String packageName();

        String mainClassName();

        List<String> vmOpts();

        List<String> args();


        record JavaConfigImpl(boolean command, boolean warnings, boolean progress, boolean verbose, String packageName,
                              String mainClassName, List<String> vmOpts, List<String> args) implements JavaConfig {
        }

        static JavaConfig of(boolean command, boolean warnings, boolean progress, boolean verbose, String packageName, String mainClassName, List<String> vmOpts, List<String> args) {
            return new JavaConfigImpl(command, warnings, progress, verbose, packageName, mainClassName, vmOpts, args);
        }

        interface Builder extends JavaConfig {
            Builder command(boolean f);

            Builder warnings(boolean f);
            Builder progress(boolean f);
            Builder verbose(boolean f);

            Builder vmOpt(String... s);

            Builder arg(String... s);

            Builder vmOpts(List<String> vmOpts);

            Builder args(List<String> vmOpts);

            Builder debug();

            Builder enablePreview();

            Builder source(int n);

            Builder addModules(String moduleName);

            Builder enableNativeAccess(String moduleName);

            Builder startOnFirstThreadIf();

            Builder startOnFirstThreadIf(boolean flag);

            Builder mainClassName(String mainClassName);

            Builder packageName(String packageName);

            Builder mainClass(String packageName, String className);

            Builder with(Consumer<Builder> builder);

            Builder whilst(Supplier<Boolean> predicate, Consumer<Builder> builder);

            Builder collectVmOpts(List<String> args);

            Builder collectArgs(List<String> args);

            class Impl implements Builder {
                boolean command;
                boolean warnings;
                boolean verbose;
                boolean progress;
                String packageName;
                String mainClassName;
                List<String> vmOpts = new ArrayList<>();
                List<String> args = new ArrayList<>();

                @Override
                public Builder command(boolean f) {
                    command = f;
                    return this;
                }

                @Override
                public Builder warnings(boolean f) {
                    warnings = f;
                    return this;
                }
                @Override
                public Builder progress(boolean f) {
                    progress = f;
                    return this;
                }

                @Override
                public Builder verbose(boolean f) {
                    verbose = f;
                    return this;
                }

                @Override
                public Builder vmOpt(String... vmOpts) {
                    this.vmOpts(List.of(vmOpts));
                    return this;
                }

                @Override
                public Builder vmOpts(List<String> vmOpts) {
                    this.vmOpts.addAll(vmOpts);
                    return this;
                }

                @Override
                public Builder arg(String... args) {
                    this.args(List.of(args));
                    return this;
                }

                @Override
                public Builder args(List<String> args) {
                    this.args.addAll(args);
                    return this;
                }

                @Override
                public Builder debug() {
                    return vmOpt("-g");
                }

                @Override
                public Builder enablePreview() {
                    return vmOpt("--enable-preview");
                }

                @Override
                public Builder startOnFirstThreadIf() {
                    return vmOpt("-XstartOnFirstThread");
                }

                @Override
                public Builder startOnFirstThreadIf(boolean flag) {
                    if (flag) {
                        startOnFirstThreadIf();
                    }
                    return this;
                }

                @Override
                public Builder source(int n) {
                    return vmOpt("--source=" + n);
                }

                @Override
                public Builder addModules(String moduleName) {
                    return vmOpt("--add-modules=" + moduleName);
                }

                @Override
                public Builder enableNativeAccess(String moduleName) {
                    return vmOpt("--enable-native-access=" + moduleName);
                }

                @Override
                public Builder mainClassName(String mainClassName) {
                    this.mainClassName = mainClassName;
                    return this;
                }

                @Override
                public Builder packageName(String packageName) {
                    this.packageName = packageName;
                    return this;
                }

                @Override
                public Builder mainClass(String packageName, String mainClassName) {
                    this.packageName = packageName;
                    this.mainClassName = packageName + "." + mainClassName;
                    return this;
                }

                @Override
                public Builder with(Consumer<Builder> nestedBuilder) {
                    nestedBuilder.accept(this);
                    return this;
                }

                @Override
                public Builder whilst(Supplier<Boolean> predicate, Consumer<Builder> nestedBuilder) {
                    while (predicate.get()) {
                        nestedBuilder.accept(this);
                    }
                    return this;
                }

                @Override
                public Builder collectVmOpts(List<String> args) {
                    return whilst(() -> (!args.isEmpty() && args.getFirst() instanceof String s && s.startsWith("-")), _ -> vmOpt(args.removeFirst()));
                }

                @Override
                public Builder collectArgs(List<String> args) {
                    return whilst(() -> !args.isEmpty(), _ -> arg(args.removeFirst()));
                }

                @Override
                public String mainClassName() {
                    return mainClassName;
                }

                @Override
                public String packageName() {
                    return packageName;
                }

                @Override
                public boolean command() {
                    return command;
                }

                @Override
                public boolean warnings() {
                    return warnings;
                }
                @Override
                public boolean progress() {
                    return progress;
                }

                @Override
                public boolean verbose() {
                    return verbose;
                }

                @Override
                public List<String> vmOpts() {
                    return vmOpts;
                }

                @Override
                public List<String> args() {
                    return args;
                }
            }
        }


        static JavaConfig of(Consumer<Builder> javaOptBuilderConsumer) {
            Builder builder = new Builder.Impl();
            javaOptBuilderConsumer.accept(builder);
            return of(builder.command(), builder.warnings(), builder.progress(), builder.verbose(), builder.packageName(), builder.mainClassName(), builder.vmOpts(), builder.args());
        }

        static JavaConfig of(JavaConfig javaOpts, Consumer<Builder> javaOptBuilderConsumer) {
            Builder builder = new Builder.Impl();
            builder
                    .verbose(javaOpts.verbose())
                    .command(javaOpts.command())
                    .packageName(javaOpts.packageName())
                    .mainClassName(javaOpts.mainClassName())
                    .vmOpts(javaOpts.vmOpts())
                    .args(javaOpts.args());
            javaOptBuilderConsumer.accept(builder);
            return of(builder.command(), builder.warnings(), builder.progress(),builder.verbose(), builder.packageName(), builder.mainClassName(), builder.vmOpts(), builder.args());
        }

        default JavaConfig with(Consumer<Builder> javaOptsBuilderConsumer){
            return JavaConfig.of(this,javaOptsBuilderConsumer);
        }
    }
}

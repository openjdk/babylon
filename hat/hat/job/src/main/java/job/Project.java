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

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class Project {
    public Dependency.Optional isAvailable(String ...nameAndArgs) {
        boolean isInPath = false;
        try {

            var process = new ProcessBuilder().command(nameAndArgs).start();
            process.waitFor();
            isInPath = (process.exitValue() == 0);
        } catch (Exception e) {
            // We'll take that as a no then  :)
        }
        return new Opt(id(nameAndArgs[0]), isInPath, Set.of());
    }

    public Path dir(String s) {
        return rootPath().resolve(s);
    }

    enum IdType {Unknown, CMakeAndJar, Jar,CMake,CMakeInfo,JExtract,Custom}

    public record Id(Project project, IdType type, String fullHyphenatedName, String projectRelativeHyphenatedName,
                     String shortHyphenatedName, String version,
                     Path path) {
        String str() {
            return project.name() + " " + fullHyphenatedName + " " + projectRelativeHyphenatedName + " " + shortHyphenatedName + " " + version + " " + (path == null ? "null" : path);
        }

        static Id of(Project project, IdType idType, String projectRelativeHyphenatedName, String shortHyphenatedName, String version, Path path) {
            return new Id(project, idType,project.name() + "-" + projectRelativeHyphenatedName + "-" + version, projectRelativeHyphenatedName, shortHyphenatedName, version, path);
        }
    }

    static Id id(Project project, String projectRelativeHyphenatedName, Path path) {

        if (projectRelativeHyphenatedName == null || projectRelativeHyphenatedName.isEmpty()) {
            throw new IllegalArgumentException("projectRelativeHyphenatedName cannot be null or empty yet");
        }
        var version = "1.0";
        if (!Files.isDirectory(path)) {
            throw new IllegalArgumentException("path "+path+" must be a directory");
        }
        int lastIndex = projectRelativeHyphenatedName.lastIndexOf('-');
        String[] names;
        if (Pattern.matches("\\d+.\\d+", projectRelativeHyphenatedName.substring(lastIndex + 1))) {
            version = projectRelativeHyphenatedName.substring(lastIndex + 1);
            names = projectRelativeHyphenatedName.substring(0, lastIndex).split("-");
        } else {
            names = projectRelativeHyphenatedName.split("-");
        }
        var tailNames = Arrays.copyOfRange(names, 1, names.length); // [] -> [....]
        var shortHyphenatedName = projectRelativeHyphenatedName;
        if (tailNames.length == 1) {
             shortHyphenatedName = tailNames[0];
        } else {
            var midNames = Arrays.copyOfRange(tailNames, 0, tailNames.length);
            shortHyphenatedName = String.join("-", midNames);
        }

        IdType idType=IdType.Custom;
        if (Files.isDirectory(path.resolve("src/main/java"))){
            if (Files.isDirectory(path.resolve("src/main/native")) && Files.isRegularFile(path.resolve("CMakeLists.txt"))){
                idType=IdType.CMakeAndJar;
            }else{
                idType=IdType.Jar;
            }
        }else  if (Files.isRegularFile(path.resolve("CMakeLists.txt"))){
            idType=IdType.CMake;
        }

        return Id.of(project, idType,projectRelativeHyphenatedName, shortHyphenatedName, version, path);
    }

    static Id id(Project project, String projectRelativeHyphenatedName) {
        var version = "1.0";
        if (projectRelativeHyphenatedName == null || projectRelativeHyphenatedName.isEmpty()) {
            throw new IllegalArgumentException("projectRelativeHyphenatedName cannot be null or empty yet");
        }
        int lastIndex = projectRelativeHyphenatedName.lastIndexOf('-');
        String[] names;
        if (Pattern.matches("\\d+.\\d+", projectRelativeHyphenatedName.substring(lastIndex + 1))) {
            version = projectRelativeHyphenatedName.substring(lastIndex + 1);
            names = projectRelativeHyphenatedName.substring(0, lastIndex).split("-");
        } else {
            names = projectRelativeHyphenatedName.split("-");
        }

        Path realPossiblyPuralizedPath = null;
        if (project.rootPath().resolve(names[0]) instanceof Path path && Files.isDirectory(path)) {
            realPossiblyPuralizedPath = path;
        } else if (project.rootPath.resolve(names[0] + "s") instanceof Path path && Files.isDirectory(path)) {
            realPossiblyPuralizedPath = path;
        }
        Id id = null;
        if (realPossiblyPuralizedPath == null || names.length == 1) {
                /* not a dir just a shortHyphenatedName or the shortHyphenatedName is a simplename (no hyphens)
                                           hyphenated                 shortHyphernated       path
                    core ->                core                       core                   <root>/core
                    mac  ->                mac                        mac                    null
                 */
            var shortHyphenatedName = projectRelativeHyphenatedName;
            id = Id.of(project, IdType.Unknown,  projectRelativeHyphenatedName, shortHyphenatedName, version, realPossiblyPuralizedPath);
        } else {
                /* we have one or more names
                                           hyphenated                 shortHyphernated       path
                    backends_ffi_opencl -> backend_ffi_opencl             ffi-opencl         <root>/backend(s)_ffi_opencl

                */
            var tailNames = Arrays.copyOfRange(names, 1, names.length); // [] -> [....]
            var expectedPath = realPossiblyPuralizedPath.resolve(String.join("/", tailNames));
            if (!Files.isDirectory(expectedPath)) {
                throw new IllegalArgumentException("The base path existed but sub path does not exist: " + expectedPath);
            } else {
                if (tailNames.length == 1) {
                    var shortHyphenatedName = tailNames[0];
                    id = Id.of(project, IdType.Unknown,projectRelativeHyphenatedName, shortHyphenatedName, version, expectedPath);
                } else {
                    var midNames = Arrays.copyOfRange(tailNames, 0, tailNames.length);
                    var shortHyphenatedName = String.join("-", midNames);
                    id = Id.of(project, IdType.Unknown,projectRelativeHyphenatedName, shortHyphenatedName, version, expectedPath);
                }
            }
        }
        return id;
    }

    public Id id(String id) {
        Pattern p = Pattern.compile("(.*)\\{([a-zA-Z0-9]+)(\\|[a-zA-Z0-9]+)?}(.*)");

        // ok lets transform the id into a project relative path
        // for example "backend{s}-ffi" -> id ="backend-ffi"  path="${project}/backends/ffi"
        if (p.matcher(id) instanceof Matcher m && m.matches() && m.groupCount() == 4) {
            id = m.group(1)+(m.group(3)==null?"":m.group(3).substring(1))+m.group(4);  // we dropped the {} and its content
            var pathName = m.group(1)+m.group(2)+m.group(4);// we included the {} content (dropped the actual braces)
            Path path =  this.rootPath.resolve(pathName.replace('-','/'));
            if (Files.isDirectory(path)) {
                //System.out.println("Id '"+id+"'->  path '"+path+"'");
                return id(id, path);
            }else{
                throw new IllegalArgumentException("Id '"+id+"' contains a path substitution but resulting path '"+path+"' does not exist");
            }
        }
        Path path = this.rootPath.resolve(this.rootPath.resolve(id.replace('-','/')));
        if (Files.isDirectory(path)) {
            //System.out.println("Id '"+id+"'->  path '"+path+"' (No substitution)");
            return  id(id, this.rootPath.resolve(path.toString()));
        }
        return id(this, id);
    }
    public Id id(String id, Path path) {
        return id(this, id, path);
    }


    private final Path rootPath;
    private final Path buildPath;
    private final Path confPath;

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

    public final Reporter reporter;

    public Project(Path root, Reporter reporter) {
        this.rootPath = root;
        if (!Files.exists(root)) {
            throw new IllegalArgumentException("Root path for project does not exist: " + root);
        }
        this.buildPath = root.resolve("build");
        this.confPath = root.resolve("conf");
        this.reporter = reporter;

    }


    public Dependency add(Dependency dependency) {
        artifacts.put(dependency.id().shortHyphenatedName, dependency);
        return dependency;
    }

    public Dependency get(String shortHyphenatedName) {
        return artifacts.get(shortHyphenatedName);
    }

    public void rmdir(Path... paths) {
        for (Path path : paths) {
            if (Files.exists(path)) {
                try (var files = Files.walk(path)) {
                    files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
                } catch (Throwable t) {
                    throw new RuntimeException(t);
                }
            }
        }
    }

    public void clean(Dependency dependency, Path... paths) {
        for (Path path : paths) {
            if (Files.exists(path)) {
                try (var files = Files.walk(path)) {
                    files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
                    reporter.command(dependency, "rm -rf " + path);
                    mkdir(path);
                    reporter.command(dependency, "mkdir -p " + path);
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

    public Dag clean(Set<Dependency> dependencies) {
        boolean all = false;
        if (dependencies.isEmpty()) {
            all = true;
            dependencies = this.artifacts.values().stream().collect(Collectors.toSet());
        }
        Dag dag = new Dag(dependencies);
        dag.ordered().stream()
                .filter(d -> d instanceof Dependency.Buildable)
                .map(d -> (Dependency.Buildable) d)
                .forEach(Dependency.Buildable::clean);
        if (all) {
            rmdir(buildPath());
        }
        return dag;
    }

    public Dag clean(String... names) {
        return clean(Set.of(names).stream().map(s -> this.artifacts.get(s)).collect(Collectors.toSet()));
    }

    public Dag build(Dag dag) {
        dag.ordered().stream()
                .filter(d -> d instanceof Dependency.Buildable)
                .map(d -> (Dependency.Buildable) d)
                .forEach(Dependency.Buildable::build);
        return dag;
    }

    public Dag build(Set<Dependency> dependencies) {
        if (dependencies.isEmpty()) {
            dependencies = this.artifacts.values().stream().collect(Collectors.toSet());
        }
        Dag dag = new Dag(dependencies);
        build(dag);
        return dag;
    }

    public Dag build(Dependency... dependencies) {
        return build(Set.of(dependencies));
    }

    public Dag all() {
        return new Dag(new HashSet<>(this.artifacts.values()));
    }
}

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

import org.w3c.dom.Element;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class Maven {
    public Path rootDir;
    public Modules modules;
    public ImlGraph imlGraph;
    public WorkspaceInfo workSpace;
    public CompilerInfo compilerInfo;

    Map<String, String> vars;

    //static Pattern varPattern = Pattern.compile("\\$\\{([^}]*)\\}");
    static Pattern varPattern = Pattern.compile("\\$([^$]*)\\$");

    public String expand(String value) { // recurse
        String result = value;
        if (varPattern.matcher(value) instanceof Matcher matcher && matcher.find()) {
            var v = matcher.group(1);
            result = expand(value.substring(0, matcher.start())
                    + (v.startsWith("env")
                    ? System.getenv(v.substring(4))
                    : vars.get(v))
                    + value.substring(matcher.end()));
            //out.println("incomming ='"+value+"'  v= '"+v+"' value='"+value+"'->'"+result+"'");
        }
        return result;
    }


    public Maven(Path rootDir) {
        this.vars = new HashMap<>();
      //  this.vars.put("PROJECT_DIR", rootDir.toString());
        this.rootDir = rootDir;
      //  var ideaDir = rootDir.resolve(".idea");
       // this.modules = new Modules(this, new XMLNode(ideaDir.resolve("modules.xml")));
       // this.imlGraph = new ImlGraph(this, modules);
       // this.workSpace = new WorkspaceInfo(this, new XMLNode(ideaDir.resolve("workspace.xml")));
       // this.compilerInfo = new CompilerInfo(this, new XMLNode(ideaDir.resolve("compiler.xml")));
    }

    interface Queryable {
        Maven maven();

        XMLNode xmlNode();

        default Stream<XMLNode> query(String xpath) {
            return xmlNode().nodes(xmlNode().xpath(xpath)).map(e -> new XMLNode((Element) e));
        }

        interface withUrl extends Queryable {
            default String url() {
                return xmlNode().attr("url");
            }
        }

        interface withType extends Queryable {
            default String type() {
                return xmlNode().attr("type");
            }
        }

        interface withPath extends Queryable {
            default String path() {
                return xmlNode().attr("path");
            }
        }
        interface withValue extends Queryable {
            default String value() {
                return xmlNode().attr("value");
            }
        }
        interface withName extends Queryable {
            default String name() {
                return xmlNode().attr("name");
            }
        }
        interface withOptions extends Queryable {
            default String[] options() {
                return xmlNode().attr("options").split(" ");
            }
        }
        interface withText extends Queryable {
            default String text(String nodeName) {
            //    return xmlNode().query(nodeName).findFirst().get().element.getTextContent();
                return xmlNode().xmlNodes(xmlNode().xpath(nodeName)).findFirst().get().element.getTextContent();
            }
            default String text() {
                //    return xmlNode().query(nodeName).findFirst().get().element.getTextContent();
                return xmlNode().element.getTextContent();
            }
        }
    }

    public record Modules(Maven maven, XMLNode xmlNode) implements Queryable {
        public record Module(Path imlPath, Maven maven, XMLNode xmlNode) implements Queryable {
            public record ModuleOrderEntry(Path imlPath, Maven maven, XMLNode xmlNode) implements withType {
            }

            public record ModuleLibraryOrderEntry(Maven maven, XMLNode xmlNode) implements withType {
                public record Library(Maven maven, XMLNode xmlNode) implements Queryable {
                    public record Classes(Maven maven, XMLNode xmlNode) implements Queryable {
                        public record Root(Maven maven, XMLNode xmlNode) implements withUrl { }
                        public Stream<Root> roots() {
                            return query("root").map(e -> new Root(maven, e));
                        }
                    }

                    public Stream<Classes> listOfClasses() {
                        return query("CLASSES").map(e -> new Classes(maven, e));
                    }
                }

                public Stream<Library> libraries() {
                    return query("library").map(e -> new Library(maven, e));
                }
            }

            public record Content(Maven maven, XMLNode xmlNode) implements withUrl {
                public record SourceFolder(Maven maven, XMLNode xmlNode) implements withUrl, withType { }

                public record ExcludeFolder(Maven maven, XMLNode xmlNode) implements withUrl { }

                public Stream<SourceFolder> sourceFolders() {
                    return query("sourceFolder").map(e -> new SourceFolder(maven, e));
                }

                public Stream<ExcludeFolder> excludeFolders() {
                    return query("excludeFolder").map(e -> new ExcludeFolder(maven, e));
                }
            }

            public Stream<ModuleOrderEntry> moduleOrderEntries() {
                return query("/module/component/orderEntry[@type='module']")
                        .map(e -> new ModuleOrderEntry(imlPath.getParent().resolve(e.attr("module-name") + ".iml"), maven, e));
            }

            public Stream<ModuleLibraryOrderEntry> moduleLibraryOrderEntries() {
                return query("/module/component/orderEntry[@type='module-library']").map(e -> new ModuleLibraryOrderEntry(maven, e));
            }

            public Stream<Content> content() {
                return query("/module/component/content").map(e -> new Content(maven, e));
            }
        }

        public Stream<Module> modules() {
            return query("/project/component[@name='ProjectModuleManager']/modules/module")
                    .map(xmlNode -> Path.of(xmlNode.attrMap.get("filepath").replace("$PROJECT_DIR$", maven().rootDir.toString())))
                    .map(path -> new Module(path, maven, new XMLNode(path)));
        }
    }

    public record WorkspaceInfo(Maven maven, XMLNode xmlNode) implements Queryable {
        public record ApplicationInfo(Maven maven, XMLNode xmlNode) implements Queryable {
            public record ClassPath(Maven maven, XMLNode xmlNode) implements Queryable {
                public record Entry(Maven maven, XMLNode xmlNode) implements withPath {
                }

                public Stream<Entry> entries() {
                    return query("entry").map(n -> new Entry(maven(), n));
                }
            }

            public String className() {
                return query("option[@name='MAIN_CLASS_NAME']").findFirst().get().attr("value");
            }

            public String[] vmArgs() {
                return query("option[@name='VM_PARAMETERS']").findFirst().get().attr("value").split(" ");
            }

            public ClassPath classPath() {
                return new ClassPath(maven(), query("classpathModifications").findFirst().get());
            }
        }

        public Stream<ApplicationInfo> applications() {
            return query("/project/component[@name='RunManager']/configuration").map(n -> new ApplicationInfo(maven(), n));
        }
    }

    public record CompilerInfo(Maven maven, XMLNode xmlNode) implements Queryable {
        public record CompilerConfiguration(Maven maven, XMLNode xmlNode) implements Queryable {
            public record ExcludeFromCompile(Maven maven, XMLNode xmlNode) implements Queryable {
                public record File(Maven maven, XMLNode xmlNode) implements withUrl {

                }
               public  Stream<File> files(){
                    return query("file").map(n -> new File(maven, n));
                }
            }
           public  Stream<ExcludeFromCompile> excludeFromCompile(){
                return query("excludeFromCompile").map(n -> new ExcludeFromCompile(maven, n));
            }
        }

        public record JavacSettings(Maven maven, XMLNode xmlNode) implements Queryable {
            public record AdditionalOptionsStrings(Maven maven, XMLNode xmlNode) implements withValue, withName {

            }

            public record AdditionalOptionsOverride(Maven maven, XMLNode xmlNode) implements Queryable {
                public record Module(Maven maven, XMLNode xmlNode) implements withName, withOptions {

                }
                public Stream<Module> modules() {
                    return query("module").map(n -> new Module(maven(), n));
                }
            }

            public Stream<AdditionalOptionsStrings> additionalOptionsStrings() {
                return query("option[@name='ADDITIONAL_OPTIONS_STRING']").map(n -> new AdditionalOptionsStrings(maven(), n));
            }

            public Stream<AdditionalOptionsOverride> additionalOptionsOverride() {
                return query("option[@name='ADDITIONAL_OPTIONS_OVERRIDE']").map(n -> new AdditionalOptionsOverride(maven(), n));
            }
        }
      public  Stream<JavacSettings> javacSettings() {
            return query("component[@name='JavacSettings']").map(n -> new JavacSettings(maven(), n));
        }
      public  Stream<CompilerConfiguration> compilerConfigurations() {
            return query("component[@name='CompilerConfiguration']").map(n -> new CompilerConfiguration(maven(), n));
        }
    }


    public static class ImlGraph {
        Maven intellij;
        Set<Modules.Module> moduleSet = new HashSet<>();
        public Map<Modules.Module, List<Modules.Module>> fromToDependencies = new HashMap<>();
        Map<Modules.Module, List<Modules.Module>> toFromDependencies = new HashMap<>();

        ImlGraph(Maven intellij, Modules modules) {
            this.intellij = intellij;
            Map<Path, Modules.Module> pathToModule = new HashMap<>();
            modules.modules().forEach(module -> {
                moduleSet.add(module);
                pathToModule.put(module.imlPath(), module);
            });
            moduleSet.forEach(module ->
                    module.moduleOrderEntries().forEach(moduleOrderEntry -> {
                                fromToDependencies.computeIfAbsent(pathToModule.get(module.imlPath()), _ -> new ArrayList<>()).add(pathToModule.get(moduleOrderEntry.imlPath));
                                toFromDependencies.computeIfAbsent(pathToModule.get(moduleOrderEntry.imlPath), _ -> new ArrayList<>()).add(pathToModule.get(module.imlPath()));
                            }
                    ));
        }
    }

    /*----*/
    public record POM(Maven maven, XMLNode xmlNode) implements Queryable {
        public record Project(Path path,Maven maven, XMLNode xmlNode) implements Queryable {
            public record Module(Maven maven, XMLNode xmlNode) implements Queryable.withText {
            }
            String modelVersion(){
                return query("modelVersion").findFirst().get().element.getTextContent();
            }
            String packaging(){
                return query("packaging").findFirst().get().element.getTextContent();
            }
            String groupId(){
                return query("groupId").findFirst().get().element.getTextContent();
            }
            String artifactId(){
                return query("artifactId").findFirst().get().element.getTextContent();
            }
            String version(){
                return query("version").findFirst().get().element.getTextContent();
            }

            public Stream<Module> modules() {
                return query("modules").map(m->new Module(maven, m));
            }
        }
        Stream<Project> projects(Path path) {
            return query("/project").map(p->new Project(path, maven, xmlNode));
        }

        public record Module(Path imlPath, Maven maven, XMLNode xmlNode) implements Queryable {
            public record ModuleOrderEntry(Path imlPath, Maven maven, XMLNode xmlNode) implements withType {
            }

            public record ModuleLibraryOrderEntry(Maven maven, XMLNode xmlNode) implements withType {
                public record Library(Maven maven, XMLNode xmlNode) implements Queryable {
                    public record Classes(Maven maven, XMLNode xmlNode) implements Queryable {
                        public record Root(Maven maven, XMLNode xmlNode) implements withUrl { }
                        public Stream<Root> roots() {
                            return query("root").map(e -> new Root(maven, e));
                        }
                    }

                    public Stream<Classes> listOfClasses() {
                        return query("CLASSES").map(e -> new Classes(maven, e));
                    }
                }

                public Stream<Library> libraries() {
                    return query("library").map(e -> new Library(maven, e));
                }
            }

            public record Content(Maven maven, XMLNode xmlNode) implements withUrl {
                public record SourceFolder(Maven maven, XMLNode xmlNode) implements withUrl, withType { }

                public record ExcludeFolder(Maven maven, XMLNode xmlNode) implements withUrl { }

                public Stream<SourceFolder> sourceFolders() {
                    return query("sourceFolder").map(e -> new SourceFolder(maven, e));
                }

                public Stream<ExcludeFolder> excludeFolders() {
                    return query("excludeFolder").map(e -> new ExcludeFolder(maven, e));
                }
            }

            public Stream<ModuleOrderEntry> moduleOrderEntries() {
                return query("/module/component/orderEntry[@type='module']")
                        .map(e -> new ModuleOrderEntry(imlPath.getParent().resolve(e.attr("module-name") + ".iml"), maven, e));
            }

            public Stream<ModuleLibraryOrderEntry> moduleLibraryOrderEntries() {
                return query("/module/component/orderEntry[@type='module-library']").map(e -> new ModuleLibraryOrderEntry(maven, e));
            }

            public Stream<Content> content() {
                return query("/module/component/content").map(e -> new Content(maven, e));
            }
        }

        public Stream<Module> modules() {
            return query("/project/component[@name='ProjectModuleManager']/modules/module")
                    .map(xmlNode -> Path.of(xmlNode.attrMap.get("filepath").replace("$PROJECT_DIR$", maven().rootDir.toString())))
                    .map(path -> new Module(path, maven, new XMLNode(path)));
        }
    }


    public static void main(String[] argArr) throws IOException, InterruptedException {
        Path userDir = Path.of("/Users/grfrost/github/babylon-grfrost-fork/hat"/*System.getProperty("user.dir")*/);
        Maven maven = new Maven(userDir);
        var root = new POM(maven,new XMLNode(userDir.resolve("pom.xml")));
        root.projects(userDir).forEach(project -> {
            System.out.println(project.groupId());
            System.out.println(project.artifactId());
            System.out.println(project.packaging());
            System.out.println(project.version());
            System.out.println(project.modelVersion());
            project.modules().forEach(m->System.out.println(m.text()));
        });
        /*
        var compilerInfo = intelliJ.compilerInfo;
        compilerInfo.javacSettings().forEach(javacSettings -> {
            javacSettings.additionalOptionsStrings().forEach(additionalOptionsStrings -> {
                System.out.println("javac (allmodules) "+additionalOptionsStrings.value());
            });
            javacSettings.additionalOptionsOverride().forEach(additionalOptionsOverride -> {
                additionalOptionsOverride.modules().forEach(module -> {
                    System.out.println("javac module "+module.name()+ " "+String.join(" ", module.options()));
                });
            });
        });
        compilerInfo.compilerConfigurations().forEach(compilerConfiguration -> {
            compilerConfiguration.excludeFromCompile().forEach(excludeFromCompile -> {
                excludeFromCompile.files().forEach(compileFile -> {
                    System.out.println("excluding "+compileFile.url());
                });
            });
        });

        var workspace = intelliJ.workSpace;
        // System.out.println(workspace);

        workspace.applications().forEach(
                a -> {
                    System.out.println("java " + String.join(" ", a.vmArgs()) + " " + a.className());
                    Maven.WorkspaceInfo.ApplicationInfo.ClassPath cp = a.classPath();
                    cp.entries().forEach(e -> {
                        System.out.println(intelliJ.expand(e.path()));
                    });

                });
        intelliJ.modules.modules().forEach(module -> {
            System.out.println("module " + module.imlPath().getFileName());
            module.content().forEach(content -> {
                System.out.println("       " + content.url());
                content.sourceFolders().forEach(sourceFolder -> {
                    System.out.println("              " + sourceFolder.url());
                });
                content.excludeFolders().forEach(sourceFolder -> {
                    System.out.println("              exclude " + sourceFolder.url());
                });
            });
            module.moduleOrderEntries().forEach(moduleDep -> {
                System.out.println("      dep " + moduleDep.imlPath().getFileName());
            });
            module.moduleLibraryOrderEntries().forEach(moduleDep -> {
                moduleDep.libraries().forEach(library -> {
                    library.listOfClasses().forEach(libraryClass -> {
                        libraryClass.roots().forEach(rootClass -> {
                            System.out.println("      dep " + rootClass.url());
                        });
                    });
                });
            });
        }); */

    }
}

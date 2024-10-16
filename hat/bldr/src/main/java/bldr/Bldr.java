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

import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.tools.Diagnostic;
import javax.tools.DiagnosticListener;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathExpressionException;
import javax.xml.xpath.XPathFactory;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static java.io.IO.println;

public class Bldr {
    public record OS(String arch, String name, String version) {
    }

    public static OS os = new OS(System.getProperty("os.arch"), System.getProperty("os.name"), System.getProperty("os.version"));

    public record Java(String version, File home) {
    }

    public static Java java = new Java(System.getProperty("java.version"), new File(System.getProperty("java.home")));

    public record User(File home, File pwd) {
    }

    public static User user = new User(new File(System.getProperty("user.home")), new File(System.getProperty("user.dir")));


    public static class XMLNode {
        org.w3c.dom.Element element;
        List<XMLNode> children = new ArrayList<>();
        Map<String, String> attrMap = new HashMap<>();

        XMLNode(org.w3c.dom.Element element) {
            this.element = element;
            this.element.normalize();
            for (int i = 0; i < this.element.getChildNodes().getLength(); i++) {
                if (this.element.getChildNodes().item(i) instanceof org.w3c.dom.Element e) {
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

        XMLNode(File file) throws Throwable {
            this(javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file).getDocumentElement());
        }

        XMLNode(URL url) throws Throwable {
            this(javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(url.openStream()).getDocumentElement());
        }

        void write(File file) throws Throwable {
            var transformer = TransformerFactory.newInstance().newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty(OutputKeys.METHOD, "xml");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
            transformer.transform(new DOMSource(element.getOwnerDocument()), new StreamResult(file));
        }

        XPathExpression xpath(String expression) throws XPathExpressionException {
            XPath xpath = XPathFactory.newInstance().newXPath();
            return xpath.compile(expression);
        }

        Node node(XPathExpression xPathExpression) throws XPathExpressionException {
            return (Node) xPathExpression.evaluate(this.element, XPathConstants.NODE);
        }

        String string(XPathExpression xPathExpression) throws XPathExpressionException {
            return (String) xPathExpression.evaluate(this.element, XPathConstants.STRING);
        }

        NodeList nodeList(XPathExpression xPathExpression) throws XPathExpressionException {
            return (NodeList) xPathExpression.evaluate(this.element, XPathConstants.NODESET);
        }
    }

    static class POM {

        static Pattern varPattern = Pattern.compile("\\$\\{([^}]*)\\}");

        static public String varExpand(Map<String, String> props, String value) { // recurse
            String result = value;
            if (varPattern.matcher(value) instanceof Matcher matcher && matcher.find()) {
                var v = matcher.group(1);
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

    public static String pathCharSeparated(List<Path> paths) {
        StringBuilder sb = new StringBuilder();
        paths.forEach(path -> {
            if (!sb.isEmpty()) {
                sb.append(File.pathSeparatorChar);
            }
            sb.append(path);
        });
        return sb.toString();
    }

    public static Path rmdir(Path path) {
        try {
            if (Files.exists(path)) {
                Files.walk(path).sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(File::delete);
            }
        } catch (IOException ioe) {
            System.out.println(ioe);
        }
        return path;
    }


    public static class JavacJarConfig {
        public Path jar;
        public List<String> opts = new ArrayList<>();
        public Path classesDir;
        public List<Path> sourcePath;
        public List<Path> classPath;
        public List<Path> resourcePath;

        public JavacJarConfig seed(JavacJarConfig stem) {
            if (stem != null) {
                if (stem.jar != null) {
                    this.jar = stem.jar;
                }
                if (stem.opts != null) {
                    this.opts = new ArrayList<>(stem.opts);
                }
                if (stem.classesDir != null) {
                    this.classesDir = stem.classesDir;
                }
                if (stem.sourcePath != null) {
                    this.sourcePath = new ArrayList<>(stem.sourcePath);
                }
                if (stem.classPath != null) {
                    this.classPath = new ArrayList<>(stem.classPath);
                }
                if (stem.resourcePath != null) {
                    this.resourcePath = new ArrayList<>(stem.resourcePath);
                }
            }
            return this;
        }

        public JavacJarConfig jar(Path jar) {
            this.jar = jar;
            return this;
        }

        public JavacJarConfig opts(List<String> opts) {
            this.opts.addAll(opts);
            return this;
        }

        public JavacJarConfig opts(String... opts) {
            opts(Arrays.asList(opts));
            return this;
        }

        public JavacJarConfig source_path(Path... sourcePaths) {
            this.sourcePath = new ArrayList<>(Arrays.asList(sourcePaths));
            return this;
        }

        public JavacJarConfig class_path(Path... classPaths) {
            this.classPath = new ArrayList<>(Arrays.asList(classPaths));
            return this;
        }

        public JavacJarConfig resource_path(Path... resourcePaths) {
            this.resourcePath = new ArrayList<>(Arrays.asList(resourcePaths));
            return this;
        }
    }

    public static JavacJarConfig javacjarconfig(Consumer<JavacJarConfig> javacJarConfigConsumer) {
        JavacJarConfig javacJarConfig = new JavacJarConfig();
        javacJarConfigConsumer.accept(javacJarConfig);
        return javacJarConfig;
    }

    public static JavacJarConfig javacjar(Consumer<JavacJarConfig> javacJarConfigConsumer) throws IOException {
        JavacJarConfig javacJarConfig = javacjarconfig(javacJarConfigConsumer);

        if (javacJarConfig.classesDir == null) {
            javacJarConfig.classesDir = javacJarConfig.jar.resolveSibling(javacJarConfig.jar.getFileName().toString() + ".classes");
        }
        javacJarConfig.opts.addAll(List.of("-d", javacJarConfig.classesDir.toString()));
        mkdir(rmdir(javacJarConfig.classesDir));

        if (javacJarConfig.classPath != null) {
            javacJarConfig.opts.addAll(List.of("--class-path", pathCharSeparated(javacJarConfig.classPath)));
        }

        javacJarConfig.opts.addAll(List.of("--source-path", pathCharSeparated(javacJarConfig.sourcePath)));
        var src = new ArrayList<Path>();
        javacJarConfig.sourcePath.forEach(entry ->
                src.addAll(paths(entry, path -> path.toString().endsWith(".java")))
        );
        if (javacJarConfig.resourcePath == null) {
            javacJarConfig.resourcePath = new ArrayList<>();
        }
        DiagnosticListener<JavaFileObject> dl = (diagnostic) -> {
            if (!diagnostic.getKind().equals(Diagnostic.Kind.NOTE)) {
                System.out.println(diagnostic.getKind()
                        + " " + diagnostic.getLineNumber() + ":" + diagnostic.getColumnNumber() + " " + diagnostic.getMessage(null));
            }
        };

        // System.out.println(builder.opts);
        record RootAndPath(Path root, Path path) {
            Path relativize() {
                return root().relativize(path());
            }
        }
        List<RootAndPath> pathsToJar = new ArrayList<>();
        JavaCompiler javac = javax.tools.ToolProvider.getSystemJavaCompiler();
        ((com.sun.source.util.JavacTask) javac.getTask(new PrintWriter(System.err), javac.getStandardFileManager(dl, null, null), dl, javacJarConfig.opts, null,
                src.stream().map(path ->
                        new SimpleJavaFileObject(path.toUri(), JavaFileObject.Kind.SOURCE) {
                            public CharSequence getCharContent(boolean ignoreEncodingErrors) {
                                try {
                                    return Files.readString(Path.of(toUri()));
                                } catch (IOException e) {
                                    throw new RuntimeException(e);
                                }
                            }
                        }).toList()
        )).generate().forEach(fileObject -> pathsToJar.add(new RootAndPath(javacJarConfig.classesDir, Path.of(fileObject.toUri()))));

        var jarStream = new JarOutputStream(Files.newOutputStream(javacJarConfig.jar));
        var setOfDirs = new HashSet<Path>();
        javacJarConfig.resourcePath.stream().sorted().forEach(resourceDir -> {
                    if (Files.isDirectory(resourceDir)) {
                        paths(resourceDir, Files::isRegularFile).forEach(path -> pathsToJar.add(new RootAndPath(resourceDir, path)));
                    }
                }
        );

        pathsToJar.stream().sorted((l, r) -> l.path().compareTo(r.path)).forEach(rootAndPath -> {
            var parentDir = rootAndPath.path().getParent();
            try {
                if (!setOfDirs.contains(parentDir)) {
                    setOfDirs.add(parentDir);
                    PosixFileAttributes attributes = Files.readAttributes(rootAndPath.path(), PosixFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
                    var entry = new JarEntry(rootAndPath.relativize() + "/");
                    entry.setTime(attributes.lastModifiedTime().toMillis());
                    jarStream.putNextEntry(entry);
                    jarStream.closeEntry();
                }
                PosixFileAttributes attributes = Files.readAttributes(rootAndPath.path(), PosixFileAttributes.class, LinkOption.NOFOLLOW_LINKS);
                var entry = new JarEntry(rootAndPath.relativize().toString());
                entry.setTime(attributes.lastModifiedTime().toMillis());
                jarStream.putNextEntry(entry);
                if (attributes.isRegularFile()) {
                    Files.newInputStream(rootAndPath.path()).transferTo(jarStream);
                }
                jarStream.closeEntry();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        jarStream.finish();
        jarStream.close();
        return javacJarConfig;
    }

    public static Path path(String name) {
        return Path.of(name);
    }

    public static Path path(Path parent, String name) {
        return parent.resolve(name);
    }

    public static List<Path> paths(Path... paths) {
        List<Path> selectedPaths = new ArrayList<>();
        Arrays.asList(paths).forEach(path -> {
            if (Files.isDirectory(path)) {
                selectedPaths.add(path);
            }
        });
        return selectedPaths;
    }

    public static List<Path> paths(Path parent, String... names) {
        List<Path> selectedPaths = new ArrayList<>();
        Arrays.asList(names).forEach(name -> {
            Path path = path(parent, name);
            if (Files.isDirectory(path)) {
                selectedPaths.add(path);
            }
        });
        return selectedPaths;
    }

    public static List<Path> paths(Path path, Predicate<Path> predicate) {
        try {
            return Files.walk(path).filter(predicate).toList();
        } catch (IOException ioe) {
            throw new IllegalStateException(ioe);
        }
    }

    public static class CMakeConfig {
        public List<String> opts = new ArrayList<>(List.of("cmake"));
        public List<String> libraries = new ArrayList<>();
        public Path cmakeBldDebugDir;
        public Path cwd;
        private String targetPackage;
        private Path output;

        public CMakeConfig seed(CMakeConfig stem) {

            if (stem != null) {
                if (stem.output != null) {
                    this.output = stem.output;
                }
                if (stem.opts != null) {
                    this.opts = new ArrayList<>(stem.opts);
                }
                if (stem.libraries != null) {
                    this.libraries = new ArrayList<>(stem.libraries);
                }
                if (stem.cwd != null) {
                    this.cwd = stem.cwd;
                }
                if (stem.cmakeBldDebugDir != null) {
                    this.cmakeBldDebugDir = stem.cmakeBldDebugDir;
                }
                if (stem.targetPackage != null) {
                    this.targetPackage = targetPackage;
                }
            }
            return this;
        }

        public CMakeConfig _B(Path cmakeBldDebugDir) {
            this.cmakeBldDebugDir = cmakeBldDebugDir;
            opts.addAll(List.of("-B", cmakeBldDebugDir.getFileName().toString()));
            return this;
        }

        public CMakeConfig __build(Path cmakeBldDebugDir) {
            this.cmakeBldDebugDir = cmakeBldDebugDir;
            opts.addAll(List.of("--build", cmakeBldDebugDir.getFileName().toString()));
            return this;
        }

        public CMakeConfig cwd(Path cwd) {
            this.cwd = cwd;
            return this;
        }

        public CMakeConfig opts(String... opts) {
            this.opts.addAll(Arrays.asList(opts));
            return this;
        }

    }

    public static CMakeConfig cmakeconfig(Consumer<CMakeConfig> cMakeConfigConsumer) {
        CMakeConfig cmakeConfig = new CMakeConfig();
        cMakeConfigConsumer.accept(cmakeConfig);
        return cmakeConfig;
    }

    public static void cmake(Consumer<CMakeConfig> cMakeConfigConsumer) {
        CMakeConfig cmakeConfig = cmakeconfig(cMakeConfigConsumer);
        try {
            Files.createDirectories(cmakeConfig.cmakeBldDebugDir);
            //System.out.println(cmakeConfig.opts);
            var cmakeProcessBuilder = new ProcessBuilder()
                    .directory(cmakeConfig.cwd.toFile())
                    .inheritIO()
                    .command(cmakeConfig.opts)
                    .start();
            cmakeProcessBuilder.waitFor();
        } catch (InterruptedException ie) {
            System.out.println(ie);
        } catch (IOException ioe) {
            System.out.println(ioe);
        }
    }

    public static class JExtractConfig {
        public List<String> opts = new ArrayList<>(List.of("jextract"));
        public List<String> compileFlags = new ArrayList<>();
        public List<Path> libraries = new ArrayList<>();
        public List<Path> headers = new ArrayList<>();
        public Path cwd;

        public Path home;
        private String targetPackage;
        private Path output;

        public JExtractConfig seed(JExtractConfig stem) {
            if (stem != null) {
                if (stem.output != null) {
                    this.output = stem.output;
                }
                if (stem.opts != null) {
                    this.opts = new ArrayList<>(stem.opts);
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


        public JExtractConfig cwd(Path cwd) {
            this.cwd = cwd;
            return this;
        }

        public JExtractConfig home(Path home) {
            this.home = home;
            opts.remove(0);
            opts.add(0, path(home, "bin/jextract").toString());
            return this;
        }

        public JExtractConfig opts(String... opts) {
            this.opts.addAll(Arrays.asList(opts));
            return this;
        }

        public JExtractConfig target_package(String targetPackage) {
            this.targetPackage = targetPackage;
            this.opts.addAll(List.of(
                    "--target-package",
                    targetPackage
            ));
            return this;
        }

        public JExtractConfig output(Path output) {
            this.output = output;
            this.opts.addAll(List.of(
                    "--output",
                    output.toString()
            ));
            return this;
        }

        public JExtractConfig library(Path... libraries) {
            this.libraries.addAll(Arrays.asList(libraries));
            for (Path library : libraries) {
                this.opts.addAll(List.of("--library", ":" + library));
            }
            return this;
        }

        public JExtractConfig l(Path... libraries) {
            return library(libraries);
        }

        public JExtractConfig compile_flag(String... compileFlags) {
            this.compileFlags.addAll(Arrays.asList(compileFlags));
            return this;
        }

        public JExtractConfig header(Path header) {
            this.headers.add(header);
            this.opts.add(header.toString());
            return this;
        }
    }

    public static JExtractConfig jextractconfig(Consumer<JExtractConfig> jextractConfigConsumer) {
        JExtractConfig extractConfig = new JExtractConfig();
        jextractConfigConsumer.accept(extractConfig);
        return extractConfig;
    }

    public static void jextract(Consumer<JExtractConfig> jextractConfigConsumer) {
        JExtractConfig extractConfig = jextractconfig(jextractConfigConsumer);
        System.out.println(extractConfig.opts);
        var compilerFlags = path(extractConfig.cwd, "compiler_flags.txt");
        try {
            PrintWriter compilerFlagsWriter = new PrintWriter(Files.newOutputStream(compilerFlags));
            compilerFlagsWriter.println(extractConfig.compileFlags);
            compilerFlagsWriter.close();

            Files.createDirectories(extractConfig.output);
            var jextractProcessBuilder = new ProcessBuilder()
                    .directory(extractConfig.cwd.toFile())
                    .inheritIO()
                    .command(extractConfig.opts)
                    .start();
            jextractProcessBuilder.waitFor();
            Files.deleteIfExists(compilerFlags);
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException ie) {
            System.out.println(ie);
        }
    }

    public static boolean existingDir(Path dir) {
        return Files.exists(dir);
    }

    public static Path mkdir(Path path) {
        try {
            return Files.createDirectories(path);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public record TextFile(Path path) {
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
    }

    public record Line(String line, int num) {
        public boolean grep(Pattern pattern) {
            return pattern.matcher(line()).matches();
        }
    }

    record GroupArtifactVersion(String group, String artifact, String version) {

    }

    interface RepoNode {
        XMLNode xmlNode();

        default GroupArtifactVersion groupArtifactVersion() {
            try {
                var groupIdXPath = xmlNode().xpath("groupId/text()");
                var group = xmlNode().string(groupIdXPath);
                var artifactIdXPath = xmlNode().xpath("artifactId/text()");
                var artifact = xmlNode().string(artifactIdXPath);
                var versionXPath = xmlNode().xpath("version/text()");
                var version = xmlNode().string(versionXPath);
                return new GroupArtifactVersion(group, artifact, version);
            } catch (XPathExpressionException xPathExpressionException) {
                throw new RuntimeException(xPathExpressionException);
            }
        }

        default String location() {
            GroupArtifactVersion groupArtifactVersion = groupArtifactVersion();
            return "https://repo1.maven.org/maven2/" + groupArtifactVersion.group().replace('.', '/') + "/" + groupArtifactVersion().artifact() + "/" + groupArtifactVersion.version();
        }

        default String name(String suffix) {
            GroupArtifactVersion groupArtifactVersion = groupArtifactVersion();
            return groupArtifactVersion.artifact() + "-" + groupArtifactVersion.version + "." + suffix;
        }

        default URL url(String suffix) {
            try {
                return new URI(location() + "/" + name(suffix)).toURL();
            } catch (MalformedURLException e) {
                throw new RuntimeException(e);
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }

        default void downloadTo(Path thirdPartyDir, String suffix) {
            var thirdPartyFile = thirdPartyDir.resolve(name(suffix));
            try {
                println("Downloading " + name(suffix) + "->" + thirdPartyDir);
                url(suffix).openStream().transferTo(Files.newOutputStream(thirdPartyFile));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    record Dependency(XMLNode xmlNode) implements RepoNode {
        public URL pomURL() {
            try {
                GroupArtifactVersion groupArtifactVersion = groupArtifactVersion();
                return new URI("https://repo1.maven.org/maven2/" + groupArtifactVersion.group().replace('.', '/') + "/" + groupArtifactVersion.artifact() + "/" + groupArtifactVersion.version() + "/"
                        + groupArtifactVersion.artifact() + "-" + groupArtifactVersion.version() + ".pom").toURL();
            } catch (MalformedURLException e) {
                throw new RuntimeException(e);
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            }
        }
    }

    record RepoPom(XMLNode xmlNode) implements RepoNode {
        List<Dependency> dependencies() {
            List<Dependency> dependencies = new ArrayList<>();
            try {
                var dependenciesXPath = xmlNode().xpath("/project/dependencies/dependency");
                var nodeList = xmlNode().nodeList(dependenciesXPath);
                for (int i = 0; i < nodeList.getLength(); i++) {
                    var node = nodeList.item(i);
                    dependencies.add(new Dependency(new XMLNode((Element) node)));
                }
                return dependencies;
            } catch (XPathExpressionException xPathExpressionException) {
                throw new RuntimeException(xPathExpressionException);
            }
        }
    }

    public static class Repo {
        Path dir;

        Repo(Path dir) {
            this.dir = dir;
        }

        Map<GroupArtifactVersion, Path> map = new HashMap<>();

        RepoPom get(String groupId, String artifactId, String version) {
            try {
                var pom = new RepoPom(new XMLNode(new URI("https://repo1.maven.org/maven2/" + groupId.replace('.', '/') + "/"
                        + artifactId + "/" + version + "/"
                        + artifactId + "-" + version + ".pom").toURL()));
                return pom;
            } catch (Throwable exception) {
                throw new RuntimeException(exception);
            }

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

    public static Path curlIfNeeded(URL url, Path file) {
        if (!Files.isRegularFile(file)) {
            curl(url, file);
        }
        return file;
    }

    public static Path untarIfNeeded(Path tarFile, Path expectedDir) {
        if (!existingDir(expectedDir)) {
            untar(tarFile, tarFile.getParent());
        }
        return expectedDir;
    }

    public static boolean available(String execName) {
        // We could just look up the env.PATH?  or we could just try to execute assuming it will need some args ;)
        try {
            new ProcessBuilder().command(execName).start().waitFor();
            return true;
        } catch (
                InterruptedException e) { // We get IOException if the executable not found, at least on Mac so interuppted means it exists
            return true;
        } catch (IOException e) { // We get IOException if the executable not found, at least on Mac
            //throw new RuntimeException(e);
            return false;
        }
    }

    public static boolean untar(Path tarFile, Path dir) {
        // We could just look up the env.PATH?  or we could just try to execute assuming it will need some args ;)
        try {
            // tar xvf thirdparty/jextract.tar --directory thirdparty
            new ProcessBuilder().inheritIO().command("tar", "xvf", tarFile.toString(), "--directory", dir.toString()).start().waitFor();
            return true;
        } catch (
                InterruptedException e) { // We get IOException if the executable not found, at least on Mac so interuppted means it exists
            return false;
        } catch (IOException e) { // We get IOException if the executable not found, at least on Mac
            //throw new RuntimeException(e);
            return false;
        }
    }

    //  https://stackoverflow.com/questions/23272861/how-to-call-testng-xml-from-java-main-method
    public static void main(String[] args) throws Throwable {
        var hatDir = path("/Users/grfrost/github/babylon-grfrost-fork/hat");
        var thirdPartyDir = path(hatDir, "thirdparty");// maybe clean?
        var repo = new Repo(thirdPartyDir);

        var jextractDir = untarIfNeeded(
                curlIfNeeded(
                        new URI("https://download.java.net/java/early_access/jextract/22/5/openjdk-22-jextract+5-33_macos-aarch64_bin.tar.gz").toURL(),
                        path(thirdPartyDir, "jextract.tar")),
                path(thirdPartyDir, "jextract-22"));


        GroupArtifactVersion g = new GroupArtifactVersion("org.testng", "testng", "7.1.0");
        GroupArtifactVersion aparapi = new GroupArtifactVersion("com.aparapi", "aparapi", "3.0.2");
        GroupArtifactVersion aparapi_jni = new GroupArtifactVersion("com.aparapi", "aparapi-jni", "1.4.3");
        GroupArtifactVersion aparapi_examples = new GroupArtifactVersion("com.aparapi", "aparapi-examples", "3.0.0");
        RepoPom testng = repo.get("org.testng", "testng", "7.1.0");

        //  var url = new URI("https://repo1.maven.org/maven2/org/testng/testng/7.1.0/testng-7.1.0.pom").toURL();
        //  var node = new XMLNode(url);
        //  RepoPom testng = new RepoPom(new XMLNode(new URI("https://repo1.maven.org/maven2/org/testng/testng/7.1.0/testng-7.1.0.pom").toURL()));
        testng.downloadTo(repo.dir, "jar");
        // testng.dependencies().stream().forEach(dependency->println(dependency.group()));
        // testng.dependencies().stream().forEach(dependency->println(dependency.pomURL()));

        // https://repo1.maven.org/maven2/org/testng/testng/7.1.0/testng-7.1.0.jar
        // var hatDir = path("/Users/grfrost/github/babylon-grfrost-fork/hat");
        var licensePattern = Pattern.compile("^.*Copyright.*202[4-9].*(Intel|Oracle).*$");
        var eolws = Pattern.compile("^.* $");
        var tab = Pattern.compile("^.*\\t.*");

        paths(hatDir, "hat", "examples", "backends").forEach(dir -> {
            paths(dir, path -> !Pattern.matches("^.*(-debug|rleparser).*$", path.toString())
                    && Pattern.matches("^.*\\.(java|cpp|h|hpp)$", path.toString())
            ).stream().map(path -> new TextFile(path)).forEach(textFile -> {
                if (!textFile.grep(licensePattern)) {
                    System.err.println("ERR MISSING LICENSE " + textFile.path());
                }
                textFile.lines().forEach(line -> {
                    if (line.grep(tab)) {
                        System.err.println("ERR TAB " + textFile.path() + ":" + line.line() + "#" + line.num());
                    }
                    if (line.grep(eolws)) {
                        System.err.println("ERR TRAILING WHITESPACE " + textFile.path() + ":" + line.line() + "#" + line.num());
                    }
                });
            });
        });

        var target = path(hatDir, "build");// mkdir(rmdir(path(hatDir, "build")));

        var hatJavacOpts = javacjarconfig($ -> $.opts(
                "--source", "24",
                "--enable-preview",
                "--add-exports=java.base/jdk.internal=ALL-UNNAMED",
                "--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED"
        ));


        var hatJarResult = javacjar($ -> $
                .seed(hatJavacOpts)
                .jar(path(target, "hat-1.0.jar"))
                .source_path(path(hatDir, "hat/src/main/java"))
        );
        var hatExampleJavaConfig = javacjarconfig($ -> $.seed(hatJavacOpts).class_path(hatJarResult.jar));
        println(hatJarResult.jar);
        for (var exampleDir : paths(path(hatDir, "examples"), "mandel", "squares", "heal", "violajones", "life")) {
            javacjar($ -> $
                    .seed(hatExampleJavaConfig)
                    .jar(path(target, "hat-example-" + exampleDir.getFileName() + "-1.0.jar"))
                    .source_path(path(exampleDir, "src/main/java"))
                    .resource_path(path(exampleDir, "src/main/resources"))
            );
        }
        var backendsDir = path(hatDir, "backends");
        for (var backendDir : paths(backendsDir, "opencl", "ptx")) {
            javacjar($ -> $
                    .seed(hatExampleJavaConfig)
                    .jar(path(target, "hat-backend-" + backendDir.getFileName() + "-1.0.jar"))
                    .source_path(path(backendDir, "src/main/java"))
                    .resource_path(path(backendDir, "src/main/resources"))
            );
        }
        var hattricksDir = path(hatDir, "hattricks");

        if (Files.exists(hattricksDir)) {
            for (var hattrickDir : paths(hattricksDir, "chess", "view")) {
                javacjar($ -> $
                        .seed(hatExampleJavaConfig)
                        .jar(path(target, "hat-example-" + hattrickDir.getFileName() + "-1.0.jar"))
                        .source_path(path(hattrickDir, "src/main/java"))
                        .resource_path(path(hattrickDir, "src/main/resources"))
                );
            }

            for (var hattrickDir : paths(hattricksDir, "nbody")) {
                var appFrameworks = "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks";
                var MAC_APP_FRAMEWORKS = Path.of(appFrameworks);
                var MAC_LIB_FRAMEWORKS = Path.of("/System/Library/Frameworks");
                var jextractedJava = path(target, "jextracted-java");
                mkdir(jextractedJava);
                var jextractedOpenCL = path(jextractedJava, "opencl");
                var jextractedOpenGL = path(jextractedJava, "opengl");
                var jextractconfig = jextractconfig($ -> $
                        .home(jextractDir)
                        .cwd(hattrickDir)
                        .output(jextractedJava)
                        .compile_flag("-F" + MAC_APP_FRAMEWORKS)
                );
                if (!existingDir(jextractedOpenCL)) {
                    jextract($ -> $
                            .seed(jextractconfig)
                            .target_package("opencl")
                            .library(path(MAC_LIB_FRAMEWORKS, "OpenCL.framework/OpenCL"))
                            .header(path(MAC_APP_FRAMEWORKS, "OpenCL.framework/Headers/opencl.h"))
                    );
                }
                if (!existingDir(jextractedOpenGL)) {
                    jextract($ -> $
                            .seed(jextractconfig)
                            .target_package("opengl")
                            .library(path(MAC_LIB_FRAMEWORKS, "GLUT.framework/GLUT"), path(MAC_LIB_FRAMEWORKS, "OpenGL.framework/OpenGL"))
                            .header(path(MAC_APP_FRAMEWORKS, "GLUT.framework/Headers/glut.h"))
                    );
                }

                javacjar($ -> $
                        .seed(hatExampleJavaConfig)
                        .jar(path(target, "hat-example-" + hattrickDir.getFileName() + "-1.0.jar"))
                        .source_path(path(hattrickDir, "src/main/java"), jextractedOpenCL, jextractedOpenGL)
                        .resource_path(path(hattrickDir, "src/main/resources"))
                );
            }
        }

        var cmakeBldDebugDir = backendsDir.resolve("bld-debug");
        if (!existingDir(cmakeBldDebugDir)) {
            mkdir(cmakeBldDebugDir);
            cmake($ -> $.cwd(backendsDir)._B(cmakeBldDebugDir).opts("-DHAT_TARGET=" + target));
        }
        cmake($ -> $.cwd(backendsDir).__build(cmakeBldDebugDir));

    }
}

package bldr;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static bldr.Bldr.curl;
import static java.io.IO.print;
import static java.io.IO.println;

public class MavenStyleRepository {
    private final String repoBase = "https://repo1.maven.org/maven2/";
    private final String searchBase = "https://search.maven.org/solrsearch/";
    public Bldr.RepoDir dir;

    Bldr.JarFile jarFile(Id id) {
        return dir.jarFile(id.artifactAndVersion() + ".jar");
    }

    Bldr.XMLFile pomFile(Id id) {
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

    public sealed interface Id permits DependencyId, MetaDataId {
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
        Bldr.JarFile getJar() {
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
                                            GroupAndArtifactId.of(
                                                    GroupId.of(dependency.xpathQueryString("groupId/text()")),
                                                    ArtifactId.of(dependency.xpathQueryString("artifactId/text()"))),
                                            VersionId.of(dependency.xpathQueryString("version/text()")),
                                            Scope.of(dependency.xpathQueryString("scope/text()")),
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

    public static class Dag implements Bldr.ClassPathEntryProvider {
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
        public List<Bldr.ClassPathEntry> classPathEntries() {
            return classPath().classPathEntries();
        }

        Bldr.ClassPath classPath() {

            Bldr.ClassPath jars = Bldr.ClassPath.of();
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

    public Bldr.ClassPathEntryProvider classPathEntries(String... rootGroupAndArtifactIds) {
        return classPathEntries(Stream.of(rootGroupAndArtifactIds).map(GroupAndArtifactId::of).toList());
    }

    public Bldr.ClassPathEntryProvider classPathEntries(GroupAndArtifactId... rootGroupAndArtifactIds) {
        return classPathEntries(List.of(rootGroupAndArtifactIds));
    }

    public Bldr.ClassPathEntryProvider classPathEntries(List<GroupAndArtifactId> rootGroupAndArtifactIds) {
      StringBuilder sb = new StringBuilder();
      rootGroupAndArtifactIds.forEach(groupAndArtifactId->sb.append(sb.isEmpty() ?"":"-").append(groupAndArtifactId.groupId+"-"+groupAndArtifactId.artifactId));
      System.out.println(sb);
      Bldr.ClassPathEntryProvider classPathEntries=null;
      var pathFileName = sb+"-path.xml";
      var pathFile = dir.xmlFile(pathFileName);
      if (pathFile.exists()){
          System.out.println(pathFileName + " exists " + pathFile.path().toString());
          XMLNode path = new XMLNode(pathFile.path());
          Bldr.ClassPath classPath = Bldr.ClassPath.of();
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

    public MavenStyleRepository(Bldr.RepoDir dir) {
        this.dir = dir.create();
    }

    Bldr.JarFile queryAndCache(URL query, Bldr.JarFile jarFile) {
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

    XMLNode queryAndCache(URL query, Bldr.XMLFile xmlFile) {
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
                                            GroupAndArtifactId.of(
                                                    GroupId.of(xmln.xpathQueryString("str[@name='g']/text()")),
                                                    ArtifactId.of(xmln.xpathQueryString("str[@name='a']/text()"))),
                                            VersionId.of(xmln.xpathQueryString("str[@name='v']/text()")),
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

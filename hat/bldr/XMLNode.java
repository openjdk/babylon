package bldr;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringWriter;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;
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

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

public class XMLNode {
  org.w3c.dom.Element element;
  List<XMLNode> children = new ArrayList<>();
  Map<String, String> attrMap = new HashMap<>();

  public static class AbstractXMLBuilder<T extends AbstractXMLBuilder<T>>{
    public org.w3c.dom.Element element;
    @SuppressWarnings("unchecked")
    public T self(){
      return (T)this;
    }
    public T attr(String name, String value) {
       // var att = element.getOwnerDocument().createAttribute(name);
        //att.setValue(value);
        element.setAttribute(name, value);
       // element.appendChild(att);
      return self();
    }
    public T attr(URI uri,String name, String value) {
      // var att = element.getOwnerDocument().createAttribute(name);
      //att.setValue(value);
      element.setAttributeNS(uri.toString(),name, value);
      // element.appendChild(att);
      return self();
    }
    public T element(String name, Function<Element,T> factory, Consumer<T> xmlBuilderConsumer) {
      var node = element.getOwnerDocument().createElement(name);
      element.appendChild(node);
      var builder = factory.apply(node);
      xmlBuilderConsumer.accept(builder);
      return self();
    }
    public T element(URI uri, String name,  Function<Element,T> factory,Consumer<T> xmlBuilderConsumer) {
      var node = element.getOwnerDocument().createElementNS(uri.toString(), name);
      element.appendChild(node);
      var builder = factory.apply(node);
      xmlBuilderConsumer.accept(builder);
      return self();
    }

    AbstractXMLBuilder(org.w3c.dom.Element element) {this.element=element;}

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
    /*
    public T oracleComment () {
      return comment("""
                        Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
                        DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.

                        This code is free software; you can redistribute it and/or modify it
                        under the terms of the GNU General Public License version 2 only, as
                        published by the Free Software Foundation.  Oracle designates this
                        particular file as subject to the "Classpath" exception as provided
                        by Oracle in the LICENSE file that accompanied this code.

                        This code is distributed in the hope that it will be useful, but WITHOUT
                        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
                        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
                        version 2 for more details (a copy is included in the LICENSE file that
                        accompanied this code).

                        You should have received a copy of the GNU General Public License version
                        2 along with this work; if not, write to the Free Software Foundation,
                        Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.

                        Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
                        or visit www.oracle.com if you need additional information or have any
                        questions.
                        """);
    } */
    <L> T forEach(List<L> list, BiConsumer<T,L> biConsumer){
      list.forEach(l->biConsumer.accept(self(),l));
      return self();
    }

    <L> T forEach(Stream<L> stream, BiConsumer<T,L> biConsumer){
      stream.forEach(l->biConsumer.accept(self(),l));
      return self();
    }

    <L> T forEach(Stream<L> stream, Consumer<L> consumer){
      stream.forEach(consumer);
      return self();
    }

    protected T then(Consumer<T> xmlBuilderConsumer) {
      xmlBuilderConsumer.accept(self());
      return self();
    }
  }
  public static class PomXmlBuilder extends AbstractXMLBuilder<PomXmlBuilder>{
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
      return element("modelVersion", $->$.text(s));
    }

    public PomXmlBuilder pom(String groupId, String artifactId, String version) {
      return modelVersion("4.0.0").packaging("pom").ref(groupId, artifactId, version);
    }
    public PomXmlBuilder jar(String groupId, String artifactId, String version) {
      return modelVersion("4.0.0").packaging("jar").ref(groupId, artifactId, version);
    }
    public PomXmlBuilder groupId(String s) {
      return element("groupId", $->$.text(s));
    }
    public PomXmlBuilder artifactId(String s) {
      return element("artifactId", $->$.text(s));
    }
    public PomXmlBuilder packaging(String s) {
      return element("packaging", $->$.text(s));
    }
    public PomXmlBuilder version(String s) {
      return element("version", $->$.text(s));
    }
    public PomXmlBuilder build(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("build", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder plugins(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("plugins", pomXmlBuilderConsumer);
    }

    public PomXmlBuilder plugin(String groupId, String artifactId, String version, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", $->$
              .ref(groupId, artifactId, version)
              .then(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder antPluginExecutions(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return plugin("org.apache.maven.plugins", "maven-antrun-plugin", "1.8", plugin -> plugin
              .executions(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder compilerPluginConfiguration(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return plugin("org.apache.maven.plugins", "maven-compiler-plugin", "3.11.0", plugin -> plugin
                                        .configuration(pomXmlBuilderConsumer)
      );
    }

    public PomXmlBuilder execPlugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return plugin("org.codehaus.mojo", "exec-maven-plugin", "3.1.0",pomXmlBuilderConsumer);
    }

    public PomXmlBuilder execPluginExecutions(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return execPlugin(plugin->plugin.executions(pomXmlBuilderConsumer));
    }

    public PomXmlBuilder plugin(String groupId, String artifactId,  Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", $->$
              .groupIdArtifactId(groupId, artifactId).then(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder plugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder parent(String groupId, String artifactId, String version){
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
    public PomXmlBuilder execIdPhaseConf(String id, String phase, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return execution(ex -> ex
              .id(id)
              .phase(phase)
              .goals(gs -> gs
                      .goal("exec")
              )
              .configuration(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder exec(String phase, String executable, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return execIdPhaseConf(executable+"-"+phase,phase,conf->conf
              .executable(executable)
              .arguments(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder cmake(String id, String phase,  Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return execIdPhaseConf(id,phase,conf->conf
              .executable("cmake")
              .arguments(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder cmake(String id, String phase,  String ...args) {
      return execIdPhaseConf(id,phase,conf->conf
              .executable("cmake")
              .arguments(arguments->arguments
                  .forEach(Stream.of(args), arguments::argument)
              )
      );
    }
    public PomXmlBuilder jextract(String id, String phase,  String ...args) {
      return execIdPhaseConf(id,phase,conf->conf
              .executable("jextract")
              .arguments(arguments->arguments
                      .forEach(Stream.of(args), arguments::argument)
              )
      );
    }

    public PomXmlBuilder ant(String id, String phase, String goal, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return execution(execution -> execution
              .id(id)
              .phase(phase)
              .goals(gs -> gs.goal(goal))
              .configuration(configuration -> configuration
                      .target(pomXmlBuilderConsumer)
              )
      );


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

    public PomXmlBuilder compilerArgs(String ...args) {
      return element("compilerArgs", $->$.forEach(Stream.of(args), $::arg));
    }
    public PomXmlBuilder properties(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("properties", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder dependencies(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("dependencies", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder dependsOn(String groupId, String artifactId, String version) {
      return element("dependencies", $->$.dependency(groupId, artifactId, version));
    }
    public    PomXmlBuilder dependency(String groupId, String artifactId, String version) {
      return dependency($->$.ref(groupId, artifactId, version));
    }
    public    PomXmlBuilder dependency(String groupId, String artifactId, String version, String scope) {
      return dependency($->$.ref(groupId, artifactId, version).scope(scope));
    }
    public PomXmlBuilder dependency(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("dependency", pomXmlBuilderConsumer);
    }

    public PomXmlBuilder modules(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("modules", pomXmlBuilderConsumer);
    }

    public PomXmlBuilder modules(String ...modules) {
       return element("modules", $->$.forEach(Stream.of(modules), $::module));
    }
    public PomXmlBuilder module(String name) {
      return element("module", $->$.text(name));
    }

    public PomXmlBuilder property(String name, String value) {
      return element(name,$->$.text(value));
    }

    public PomXmlBuilder scope(String s) {
      return element("scope", $->$.text(s));
    }
    public PomXmlBuilder phase(String s) {
      return element("phase", $->$.text(s));
    }
    public PomXmlBuilder argument(String s) {
      return element("argument", $->$.text(s));
    }

    public PomXmlBuilder goal(String s) {
      return element("goal", $->$.text(s));
    }

    public PomXmlBuilder copy(String file, String toDir) {
      return element("copy", $->$.attr("file", file).attr("toDir", toDir));
    }
    public PomXmlBuilder echo(String message) {
      return element("echo", $->$.attr("message", message));
    }
    public PomXmlBuilder echo(String filename, String message) {
      return element("echo", $->$.attr("message", message).attr("file", filename));
    }

    public PomXmlBuilder groupIdArtifactId(String groupId, String artifactId) {
      return groupId(groupId).artifactId(artifactId);
    }
    public PomXmlBuilder ref(String groupId, String artifactId, String version) {
      return groupIdArtifactId(groupId,artifactId).version(version);
    }

    public PomXmlBuilder skip(String string) {
      return element("skip", $->$.text(string));
    }

    public PomXmlBuilder id(String s) {
      return element("id", $->$.text(s));
    }

    public PomXmlBuilder arg(String s) {
      return element("arg", $->$.text(s));
    }
    public PomXmlBuilder argLine(String s) {
      return element("argLine", $->$.text(s));
    }
    public PomXmlBuilder source(String s) {
      return element("source", $->$.text(s));
    }
    public PomXmlBuilder target(String s) {
      return element("target", $->$.text(s));
    }
    public PomXmlBuilder showWarnings(String s) {
      return element("showWarnings", $->$.text(s));
    }
    public PomXmlBuilder showDeprecation(String s) {
      return element("showDeprecation", $->$.text(s));
    }
    public PomXmlBuilder failOnError(String s) {
      return element("failOnError", $->$.text(s));
    }
    public PomXmlBuilder exists(String s) {
      return element("exists", $->$.text(s));
    }
    public PomXmlBuilder activeByDefault(String s) {
      return element("activeByDefault", $->$.text(s));
    }

    public PomXmlBuilder executable(String s) {
      return element("executable", $->$.text(s));
    }
  }
  public static class ImlBuilder extends AbstractXMLBuilder<ImlBuilder>{
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
      return element("modelVersion", $->$.text(s));
    }
    public ImlBuilder groupId(String s) {
      return element("groupId", $->$.text(s));
    }
    public ImlBuilder artifactId(String s) {
      return element("artifactId", $->$.text(s));
    }
    public ImlBuilder packaging(String s) {
      return element("packaging", $->$.text(s));
    }
    public ImlBuilder version(String s) {
      return element("version", $->$.text(s));
    }
    public ImlBuilder build(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("build", pomXmlBuilderConsumer);
    }
    public ImlBuilder plugins(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("plugins", pomXmlBuilderConsumer);
    }

    public ImlBuilder plugin(String groupId, String artifactId, String version, Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", $->$
              .groupIdArtifactIdVersion(groupId, artifactId, version).then(pomXmlBuilderConsumer)
      );
    }
    public ImlBuilder plugin(String groupId, String artifactId,  Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", $->$
              .groupIdArtifactId(groupId, artifactId).then(pomXmlBuilderConsumer)
      );
    }
    public ImlBuilder plugin(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", pomXmlBuilderConsumer);
    }
    public ImlBuilder parent(String groupId, String artifactId, String version){
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
    public    ImlBuilder dependency(String groupId, String artifactId, String version) {
      return dependency($->$.groupIdArtifactIdVersion(groupId, artifactId, version));
    }
    public    ImlBuilder dependency(String groupId, String artifactId, String version, String scope) {
      return dependency($->$.groupIdArtifactIdVersion(groupId, artifactId, version).scope(scope));
    }
    public ImlBuilder dependency(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("dependency", pomXmlBuilderConsumer);
    }

    public ImlBuilder modules(Consumer<ImlBuilder> pomXmlBuilderConsumer) {
      return element("modules", pomXmlBuilderConsumer);
    }
    public ImlBuilder module(String name) {
      return element("module", $->$.text(name));
    }

    public ImlBuilder property(String name, String value) {
      return element(name,$->$.text(value));
    }

    public ImlBuilder scope(String s) {
      return element("scope", $->$.text(s));
    }
    public ImlBuilder phase(String s) {
      return element("phase", $->$.text(s));
    }
    public ImlBuilder argument(String s) {
      return element("argument", $->$.text(s));
    }

    public ImlBuilder goal(String s) {
      return element("goal", $->$.text(s));
    }

    public ImlBuilder copy(String file, String toDir) {
      return element("copy", $->$.attr("file", file).attr("toDir", toDir));
    }

    public ImlBuilder groupIdArtifactId(String groupId, String artifactId) {
        return groupId(groupId).artifactId(artifactId);
    }
    public ImlBuilder groupIdArtifactIdVersion(String groupId, String artifactId, String version) {
      return groupIdArtifactId(groupId,artifactId).version(version);
    }

    public ImlBuilder skip(String string) {
      return element("skip", $->$.text(string));
    }

    public ImlBuilder id(String s) {
      return element("id", $->$.text(s));
    }

    public ImlBuilder arg(String s) {
      return element("arg", $->$.text(s));
    }
    public ImlBuilder argLine(String s) {
      return element("argLine", $->$.text(s));
    }
    public ImlBuilder source(String s) {
      return element("source", $->$.text(s));
    }
    public ImlBuilder target(String s) {
      return element("target", $->$.text(s));
    }
    public ImlBuilder showWarnings(String s) {
      return element("showWarnings", $->$.text(s));
    }
    public ImlBuilder showDeprecation(String s) {
      return element("showDeprecation", $->$.text(s));
    }
    public ImlBuilder failOnError(String s) {
      return element("failOnError", $->$.text(s));
    }
    public ImlBuilder exists(String s) {
      return element("exists", $->$.text(s));
    }
    public ImlBuilder activeByDefault(String s) {
      return element("activeByDefault", $->$.text(s));
    }

    public ImlBuilder executable(String s) {
      return element("executable", $->$.text(s));
    }
  }
  public static class XMLBuilder extends AbstractXMLBuilder<XMLBuilder>{

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
  static XMLNode create( String nodeName, Consumer<XMLBuilder> xmlBuilderConsumer) {

      try {
          var doc  = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
          //var nl = doc.createTextNode("\n");
          //doc.appendChild(nl);
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
      var doc  = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

      var uri1 = URI.create("http://maven.apache.org/POM/4.0.0");
      var uri2 = URI.create("http://www.w3.org/2001/XMLSchema-instance");
      var uri3 = URI.create("http://maven.apache.org/xsd/maven-4.0.0.xsd");
      var comment = doc.createComment(commentText);
      doc.appendChild(comment);
      //   var nl = doc.createTextNode("\n");
      //   doc.appendChild(nl);
      var element = doc.createElementNS(uri1.toString(),"project");
      doc.appendChild(element);
      element.setAttributeNS(uri2.toString(), "xsi:schemaLocation",uri1+" "+ uri3);
      ImlBuilder imlBuilder = new ImlBuilder(element);
      imlBuilderConsumer.accept(imlBuilder);
      return new XMLNode(element);
    } catch (ParserConfigurationException e) {
      throw new RuntimeException(e);
    }
  }

  public static XMLNode createPom(String commentText, Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
    try {
      var doc  = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();

      var uri1 = URI.create("http://maven.apache.org/POM/4.0.0");
      var uri2 = URI.create("http://www.w3.org/2001/XMLSchema-instance");
      var uri3 = URI.create("http://maven.apache.org/xsd/maven-4.0.0.xsd");
      var comment = doc.createComment(commentText);
      doc.appendChild(comment);
   //   var nl = doc.createTextNode("\n");
   //   doc.appendChild(nl);
      var element = doc.createElementNS(uri1.toString(),"project");
      doc.appendChild(element);
      element.setAttributeNS(uri2.toString(), "xsi:schemaLocation",uri1+" "+ uri3);
      PomXmlBuilder pomXmlBuilder = new PomXmlBuilder(element);
      pomXmlBuilderConsumer.accept(pomXmlBuilder);
      return new XMLNode(element);
    } catch (ParserConfigurationException e) {
      throw new RuntimeException(e);
    }
  }

  static XMLNode create(URI uri, String nodeName, Consumer<XMLBuilder> xmlBuilderConsumer) {

    try {
      var doc  = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
      //var nl = doc.createTextNode("\n");
      //doc.appendChild(nl);
      var element = doc.createElementNS(uri.toString(),nodeName);
      doc.appendChild(element);
      XMLBuilder xmlBuilder = new XMLBuilder(element);
      xmlBuilderConsumer.accept(xmlBuilder);
      return new XMLNode(element);
    } catch (ParserConfigurationException e) {
      throw new RuntimeException(e);
    }



  }


  XMLNode(org.w3c.dom.Element element) {
    this.element = element;
    this.element.normalize();
    NodeList nodeList = element.getChildNodes();
    for (int i = 0; i < nodeList.getLength(); i++) {
      if (nodeList.item(i) instanceof org.w3c.dom.Element e) {
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
      return javax.xml.parsers.DocumentBuilderFactory.newInstance()
              .newDocumentBuilder()
              .parse(is);
    }catch (ParserConfigurationException | SAXException | IOException e) {
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

  XMLNode(File file)  {
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
  //  transformer.setOutputProperty("http://www.oracle.com/xml/is-standalone", "yes");
    transformer.transform(new DOMSource(element.getOwnerDocument()), streamResult);
  }

  void write(File file) {
    try {
      write(new StreamResult(file));
    }catch (Throwable t){
      throw new RuntimeException(t);
    }
  }
  public void write(Bldr.XMLFile xmlFile) {
    try {
      write(new StreamResult(xmlFile.path().toFile()));
    }catch (Throwable t){
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

  Stream<org.w3c.dom.Element> elements(XPathExpression xPathExpression) {
    return nodes(xPathExpression)
        .filter(n -> n instanceof org.w3c.dom.Element)
        .map(n -> (Element) n);
  }

  Stream<XMLNode> xmlNodes(XPathExpression xPathExpression) {
    return elements(xPathExpression).map(e -> new XMLNode(e));
  }
}

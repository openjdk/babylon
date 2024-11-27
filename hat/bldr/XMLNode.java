package bldr;

import java.io.File;
import java.io.StringWriter;
import java.net.URI;
import java.net.URL;
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
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XMLNode {
  org.w3c.dom.Element element;
  List<XMLNode> children = new ArrayList<>();
  Map<String, String> attrMap = new HashMap<>();

  public static class AbstractXMLBuilder<T extends AbstractXMLBuilder<T>>{
    public org.w3c.dom.Element element;
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
    }
    <L> T forEach(List<L> list, BiConsumer<T,L> biConsumer){
      list.forEach(l->biConsumer.accept(self(),l));
      return self();
    }

    <L> T forEach(Stream<L> stream, BiConsumer<T,L> biConsumer){
      stream.forEach(l->biConsumer.accept(self(),l));
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
              .groupIdArtifactIdVersion(groupId, artifactId, version).then(pomXmlBuilderConsumer)
      );
    }
    public PomXmlBuilder plugin(String groupId, String artifactId,  Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("plugin", $->$
              .groupIdArtifactId(groupId, artifactId).then(pomXmlBuilderConsumer)
      );
    }
  //  public PomXmlBuilder plugin(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
   //   return element("plugin", pomXmlBuilderConsumer);
  //  }
    public PomXmlBuilder parent(String groupId, String artifactId, String version){
      return parent(parent -> parent.groupIdArtifactIdVersion(groupId, artifactId, version));
    }
    public PomXmlBuilder parent(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("parent", pomXmlBuilderConsumer);
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
    public PomXmlBuilder goals(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("goals", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder target(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("target", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder configuration(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("configuration", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder properties(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("properties", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder dependencies(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("dependencies", pomXmlBuilderConsumer);
    }
    public    PomXmlBuilder dependency(String groupId, String artifactId, String version) {
      return dependency($->$.groupIdArtifactIdVersion(groupId, artifactId, version));
    }
    public PomXmlBuilder dependency(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("dependency", pomXmlBuilderConsumer);
    }

    public PomXmlBuilder modules(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
      return element("modules", pomXmlBuilderConsumer);
    }
    public PomXmlBuilder module(String name) {
      return element("module", $->$.text(name));
    }

    public PomXmlBuilder property(String name, String value) {
      return element(name,$->$.text(value));
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

    public PomXmlBuilder groupIdArtifactId(String groupId, String artifactId) {
        return groupId(groupId).artifactId(artifactId);
    }
    public PomXmlBuilder groupIdArtifactIdVersion(String groupId, String artifactId, String version) {
      return groupIdArtifactId(groupId,artifactId).version(version);
    }

    public PomXmlBuilder skip(String string) {
      return element("skip", $->$.text(string));
    }

    public PomXmlBuilder id(String s) {
      return element("id", $->$.text(s));
    }

    public PomXmlBuilder executable(String s) {
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
  static XMLNode createPom(Consumer<PomXmlBuilder> pomXmlBuilderConsumer) {
    try {
      var doc  = javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
      //var nl = doc.createTextNode("\n");
      //doc.appendChild(nl);
      var uri1 = URI.create("http://maven.apache.org/POM/4.0.0");
      var uri2 = URI.create("http://www.w3.org/2001/XMLSchema-instance");
      var uri3 = URI.create("http://maven.apache.org/xsd/maven-4.0.0.xsd");
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

  XMLNode(Path path) throws Throwable {
    this(
        javax.xml.parsers.DocumentBuilderFactory.newInstance()
            .newDocumentBuilder()
            .parse(path.toFile())
            .getDocumentElement());
  }

  XMLNode(File file) throws Throwable {
    this(
        javax.xml.parsers.DocumentBuilderFactory.newInstance()
            .newDocumentBuilder()
            .parse(file)
            .getDocumentElement());
  }

  XMLNode(URL url) throws Throwable {
    this(
        javax.xml.parsers.DocumentBuilderFactory.newInstance()
            .newDocumentBuilder()
            .parse(url.openStream())
            .getDocumentElement());
  }

  void write(StreamResult streamResult) throws Throwable {
    var transformer = TransformerFactory.newInstance().newTransformer();
    transformer.setOutputProperty(OutputKeys.INDENT, "yes");
    transformer.setOutputProperty(OutputKeys.METHOD, "xml");
    transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
    transformer.transform(new DOMSource(element.getOwnerDocument()), streamResult);
  }

  void write(File file) {
    try {
      write(new StreamResult(file));
    }catch (Throwable t){
      throw new RuntimeException(t);
    }
  }
  void write(Bldr.XMLFile xmlFile) {
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

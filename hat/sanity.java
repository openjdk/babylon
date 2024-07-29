//usr/bin/env jshell  "$0" "$@"; exit $?
//usr/bin/env jshell --execution local "$0" "$@"; exit $?

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

import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.nio.file.*;
import java.util.stream.Stream;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
public class PomChecker {
   public static class XMLNode {
      org.w3c.dom.Element element;
      List<XMLNode> children = new ArrayList<>();
      Map<String, String> attrMap =  new HashMap<>();

      XMLNode(org.w3c.dom.Element element) {
         this.element = element;
         this.element.normalize();
         for (int i = 0; i < this.element.getChildNodes().getLength(); i++) {
            if (this.element.getChildNodes().item(i) instanceof org.w3c.dom.Element e){
               this.children.add(new XMLNode(e));
            }
         }
         for (int i = 0; i < element.getAttributes().getLength(); i++) {
            if (element.getAttributes().item(i) instanceof org.w3c.dom.Attr attr){
               this.attrMap.put(attr.getName(),attr.getValue());
            }
         }
      }
      public boolean hasAttr(String name) { return attrMap.containsKey(name); }
      public String attr(String name) { return attrMap.get(name); }
      XMLNode(File file) throws Throwable {
         this(javax.xml.parsers.DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(file).getDocumentElement());
      }
      void write(File file) throws Throwable {
         var  transformer = TransformerFactory.newInstance().newTransformer();
         transformer.setOutputProperty(OutputKeys.INDENT, "yes");
         transformer.setOutputProperty(OutputKeys.METHOD, "xml");
         transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");
         transformer.transform(new DOMSource(element.getOwnerDocument()), new StreamResult(file));
      }
   }
   static Pattern varPattern=Pattern.compile("\\$\\{([^}]*)\\}");
   static Pattern trailingWhitespacePattern=Pattern.compile(".*  *");
   static public String varExpand(Map<String,String> props, String value){ // recurse
      String result = value;
      if (varPattern.matcher(value) instanceof Matcher matcher && matcher.find()) {
         var v = matcher.group(1);
         result = varExpand(props,value.substring(0, matcher.start())
               +(v.startsWith("env")
                  ?System.getenv(v.substring(4))
                  :props.get(v))
               +value.substring(matcher.end()));
         //out.println("incomming ='"+value+"'  v= '"+v+"' value='"+value+"'->'"+result+"'");
      }
      return result;
   }
   static boolean isParent(File possibleParent, File maybeChild){
      File parent = maybeChild.getParentFile();
      while ( parent != null ) {
         if ( parent.equals( possibleParent ) )
            return true;
         parent = parent.getParentFile();
      }
      return false;
   }



   public static void main(String[] args) throws Throwable{
      var out = System.out;
      var err = System.out;

      var osArch = System.getProperty("os.arch");
      var osName = System.getProperty("os.name");
      var osVersion = System.getProperty("os.version");
      var javaVersion = System.getProperty("java.version");
      var javaHome = System.getProperty("java.home");
      var pwd = new File(System.getProperty("user.dir"));

      if (javaVersion.startsWith("24")){
         //out.println("javaVersion "+javaVersion+" looks OK");

         var props = new LinkedHashMap<String,String>();
         var dir = new File(".");
         var topPom = new XMLNode(new File(dir,"pom.xml"));
         var babylonDirKey = "babylon.dir";
         var spirvDirKey = "beehive.spirv.toolkit.dir";
         var hatDirKey = "hat.dir";
         var interestingKeys = Set.of(spirvDirKey, babylonDirKey,hatDirKey);
         var requiredDirKeys = Set.of(babylonDirKey, hatDirKey);
         var dirKeyToDirMap = new HashMap<String,File>();

         topPom.children.stream().filter(e->e.element.getNodeName().equals("properties")).forEach(properties ->
               properties.children.stream().forEach(property ->{
                  var key = property.element.getNodeName();
                  var value = varExpand(props,property.element.getTextContent());
                  props.put(key, value);
                  if (interestingKeys.contains(key)){
                      var file = new File(value);
                      if (requiredDirKeys.contains(key) && !file.exists()){
                         err.println("ERR pom.xml has property '"+key+"' with value '"+value+"' but that dir does not exists!");
                         System.exit(1);
                      }
                      dirKeyToDirMap.put(key,file);
                  }
                  })
               );
         for (var key:requiredDirKeys){
             if (!props.containsKey(key)){
                 err.println("ERR pom.xml expected to have property '"+key+"' ");
                 System.exit(1);
             }
         }

         var javaHomeDir = new File(javaHome);
         var babylonDir = dirKeyToDirMap.get(babylonDirKey);
         if (isParent(babylonDir, javaHomeDir)){
            var hatDir = dirKeyToDirMap.get(hatDirKey);
            if (hatDir.equals(pwd)){
               var backendsPom = new XMLNode(new File(dir,"backends/pom.xml"));
               var modules = backendsPom.children.stream().filter(e->e.element.getNodeName().equals("modules")).findFirst().get();
               var spirvModule = modules.children.stream().filter(e->e.element.getTextContent().equals("spirv")).findFirst();
               if (spirvModule.isPresent()){
                  if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                     var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                     if (!spirvDir.exists()) {
                        err.println("ERR "+spirvDirKey + " -> '" + spirvDir + "' dir does not exists but module included in backends ");
                        System.exit(1);
                     }
                  }else{
                     err.println("ERR "+spirvDirKey + " -> variable dir does not exists but module included in backends ");
                     System.exit(1);
                  }
               } else{
                  if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                     var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                     if (spirvDir.exists()){
                        out.println("WRN "+spirvDirKey+" -> '"+spirvDir+"' exists but spirv module not included in backends ");
                     }else{
                        out.println("INF "+spirvDirKey+" -> '"+spirvDir+"' does not exist and not included in backends ");
                     }
                  }
               }
            } else{
               err.println("ERR hat.dir='"+hatDir+"' != ${pwd}='"+pwd+"'");
               System.exit(1);
            }
         }else{
            err.println("ERR babylon.dir '"+babylonDir+"' is not a child of javaHome '"+javaHome+"'");
            System.exit(1);
         }
      }else{
         err.println("ERR Incorrect Java version. Is babylon jdk in your path?");
         System.exit(1);
      }

      Set.of("hat", "examples", "backends").forEach(dirName->{
         try{
            Files.walk(Paths.get(dirName)).filter(p->{
              var name = p.toString();
              return !name.contains("cmake-build-debug")
                && !name.contains("rleparser")
                && ( name.endsWith(".java") || name.endsWith(".cpp") || name.endsWith(".h"));
              }).forEach(path->{
                try{
                   boolean license = false;
                   for (String line: Files.readAllLines(path,  StandardCharsets.UTF_8)){
                      if (line.contains("\t")){
                        err.println("ERR TAB "+path+":"+line);
                      }
                      if (line.endsWith(" ")) {
                        err.println("ERR TRAILING WHITESPACE "+path+":"+line);
                      }
                      if (Pattern.matches("^  *(package|import).*$",line)) { // I saw this a few times....?
                        err.println("ERR WEIRD INDENT "+path+":"+line);
                      }
                      if (Pattern.matches("^.*Copyright.*202[4-9].*Oracle.*$",line)) { // not foolproof I know
                        license = true;
                      }
                   }
                   if (!license){
                      err.println("ERR MISSING LICENSE "+path);
                   }
                } catch(IOException ioe){
                  err.println(ioe);
                }
            });
         } catch(IOException ioe){
           err.println(ioe);
         }
      });
   }
}


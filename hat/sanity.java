//usr/bin/env jshell  "$0" "$@"; exit $?
//usr/bin/env jshell --execution local "$0" "$@"; exit $?
import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.util.*;
import java.util.regex.*;
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
         out.println("javaVersion "+javaVersion+" looks OK");

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
                         out.println("pom.xml has property '"+key+"' with value '"+value+"' but that dir does not exists! BAD");
                         System.exit(1);
                      }
                      dirKeyToDirMap.put(key,file);
                  }
                  })
               );
         for (var key:requiredDirKeys){
             if (!props.containsKey(key)){
                 out.println("pom.xml expected to have property '"+key+"' ");
                 System.exit(1);
             }
         }

         var javaHomeDir = new File(javaHome);
         var babylonDir = dirKeyToDirMap.get(babylonDirKey);
         if (isParent(babylonDir, javaHomeDir)){
            out.println("babylon.dir '"+babylonDir+"' is parent of JAVA_HOME OK");

            var hatDir = dirKeyToDirMap.get(hatDirKey);
            if (hatDir.equals(pwd)){
               out.println("hat.dir='"+hatDir+"' OK");
               var backendsPom = new XMLNode(new File(dir,"backends/pom.xml"));
               var modules = backendsPom.children.stream().filter(e->e.element.getNodeName().equals("modules")).findFirst().get();
               var spirvModule = modules.children.stream().filter(e->e.element.getTextContent().equals("spirv")).findFirst();

               if (spirvModule.isPresent()){

                  if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                     var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                     if (spirvDir.exists()) {
                        out.println("OK "+spirvDirKey + " -> '" + spirvDir + "' dir exists and module included in backends");
                     } else {
                        out.println("ERR "+spirvDirKey + " -> '" + spirvDir + "' dir does not exists but module included in backends ");
                     }
                  }else{
                     out.println("ERR "+spirvDirKey + " -> variable dir does not exists but module included in backends ");
                  }
               } else{
                  if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                     var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                     if (spirvDir.exists()){
                        out.println("ERR "+spirvDirKey+" -> '"+spirvDir+"' dir exists but spirv module not included in backends ");
                     }else{
                        out.println("WARN "+spirvDirKey+" -> '"+spirvDir+"' dir does not exist and not included in backends ");
                     }
                  }else{
                     out.println("OK "+ spirvDirKey + " -> variable dir does not exist and module not included in backends ");
                  }
               }
            } else{
               out.println("hat.dir='"+hatDir+"' != ${pwd}='"+pwd+"' BAD");
            }
         }else{
            out.println("babylon.dir '"+babylonDir+"' is not a child of javaHome '"+javaHome+"' BAD");
         }
      }else{
         err.println("Incorrect Java version. Is babylon jdk in your path? BAD");
      }
   }
}


package experiments;

import javax.xml.transform.OutputKeys;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.util.*;
import java.util.regex.*;
public class PomChecker {
// XML facade to offer modern access to org.w3x.dom artifacts

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

    static String varExpand(Map<String,String> props, String value){ // recurse
        String result = value;
        if (varPattern.matcher(value) instanceof Matcher matcher && matcher.find()) {
            var v = matcher.group(1);
            result = varExpand(props,value.substring(0, matcher.start())
                    +(v.startsWith("env")
                    ?System.getenv(v.substring(4))
                    :props.get(v))
                    +value.substring(matcher.end()));
            //System.out.println("incomming ='"+value+"'  v= '"+v+"' value='"+value+"'->'"+result+"'");
        }
        return result;
    }

    public static void main(String[] args) throws Throwable{
         var props = new LinkedHashMap<String,String>();

        var dir = new File("/Users/grfrost/github/babylon-grfrost-fork/hat");

        var topPom = new XMLNode(new File(dir,"pom.xml"));
        topPom.write( new File(dir,"gramminet.xml"));
        topPom.children.stream()
                .filter(e->e.element.getNodeName().equals("properties")).
                forEach(properties ->
                        properties .children.stream() .forEach(property ->
                                props.put(property.element.getNodeName(),varExpand(props,property.element.getTextContent()))
                        )
                );
        //props.forEach((k,v)->System.out.println(k+"->"+v));
        var spirvDirKey = "beehive.spirv.toolkit.dir";
        var hatDirKey = "hat.dir";
        var dirKeys = new String[]{spirvDirKey, hatDirKey};
        var requiredDirKeys = new String[]{hatDirKey};
        var dirKeyToDirMap = new HashMap<String,File>();
        for (var dirKey:dirKeys){
            if (props.containsKey(dirKey)){
                dirKeyToDirMap.put(dirKey,new File(props.get(dirKey)));
            }
        }
        for (var dirKey:requiredDirKeys){
            if (props.containsKey(dirKey)){
                if (!dirKeyToDirMap.get(dirKey).exists()){
                    System.out.println("pom.xml has key'"+dirKey+"' but dir does not exists");
                    System.exit(1);
                }
            }
        }

        var hatDir = dirKeyToDirMap.get(hatDirKey);
        var hereDir = new File(System.getProperty("user.dir"));
        if (!hatDir.equals(hereDir)){
            System.out.println("hat.dir='"+hatDir+"' != ${PWD}='"+hereDir+"'");
        } else{
            System.out.println("hat.dir='"+hatDir+"' looks good");
            var backendsPom = new XMLNode(new File(dir,"backends/pom.xml"));
            var modules = backendsPom.children.stream().filter(e->e.element.getNodeName().equals("modules")).findFirst().get();
            var spirvModule = modules.children.stream().filter(e->e.element.getTextContent().equals("spirv")).findFirst();

            if (spirvModule.isPresent()){

                if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                    var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                    if (spirvDir.exists()) {
                        System.out.println("OK "+spirvDirKey + " -> '" + spirvDir + "' dir exists and module included in backends");
                    } else {
                        System.out.println("ERR "+spirvDirKey + " -> '" + spirvDir + "' dir does not exists but module included in backends ");
                    }
                }else{
                    System.out.println("ERR "+spirvDirKey + " -> variable dir does not exists but module included in backends ");
                }
            } else{
                if (dirKeyToDirMap.containsKey(spirvDirKey)) {
                    var spirvDir = dirKeyToDirMap.get(spirvDirKey);
                if (spirvDir.exists()){
                    System.out.println("ERR "+spirvDirKey+" -> '"+spirvDir+"' dir exists but spirv module not included in backends ");
                }else{
                    System.out.println("WARN "+spirvDirKey+" -> '"+spirvDir+"' dir does not exist and not included in backends ");
                }
                }else{
                    System.out.println("OK "+ spirvDirKey + " -> variable dir does not exist and module not included in backends ");
                }
            }
        }
    }

}

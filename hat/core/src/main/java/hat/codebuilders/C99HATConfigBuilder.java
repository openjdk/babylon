package hat.codebuilders;

import hat.Config;

public class C99HATConfigBuilder extends HATCodeBuilder<C99HATConfigBuilder> {

    C99HATConfigBuilder staticConstInt(String name, int padWidth, int value) {
        staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intHexValue(value).semicolon().nl();
        return this;
    }

    public C99HATConfigBuilder staticConstIntShiftedOne(String name, int padWidth, int shift) {
        staticKeyword().space().constexprKeyword().space().intType().space().identifier(name, padWidth).space().equals().space().intValue(1).leftShift().intHexValue(shift).semicolon().nl();
        return this;
    }

    public C99HATConfigBuilder className() {
        return identifier("BasicConfig");
    }

    public C99HATConfigBuilder bitNamesVar() {
        return identifier("bitNames");
    }

    public C99HATConfigBuilder bitDescriptionsVar() {
        return identifier("bitDescriptions");
    }

    public C99HATConfigBuilder configBitsVar() {
        return identifier("configBits");
    }

    public C99HATConfigBuilder configBitsAnd() {
        return configBitsVar().space().ampersand().space();
    }

    public C99HATConfigBuilder configBitsAndBitName(String bitName) {
        return configBitsAnd().identifier(bitName + "_BIT");
    }

    public static String toCamelExceptFirst(String s) {
        String[] parts = s.split("_");
        StringBuilder camelCaseString = new StringBuilder("");
        for (String part : parts) {
            camelCaseString.append(camelCaseString.isEmpty()
                    ? part.toLowerCase()
                    : part.substring(0, 1).toUpperCase() + part.substring(1).toLowerCase());
        }
        return camelCaseString.toString();
    }

    public C99HATConfigBuilder camelExceptFirst(String s) {
        return identifier(toCamelExceptFirst(s));
    }

    C99HATConfigBuilder std(String s) {
        return identifier("std").colon().colon().identifier(s);
    }

    public C99HATConfigBuilder stdEndl() {
        return std("endl");
    }

    public C99HATConfigBuilder stdCout(String s) {
        return std("cout").space().leftShift().space().dquote().literal(s).dquote();
    }

    static public void main(){
        var c = Config.of("INFO,SHOW_CODE,SHOW_KERNEL_MODEL,SHOW_COMPUTE_MODEL,PLATFORM:0,DEVICE:0");
        System.out.println(c);
        System.exit(1);
    }
}

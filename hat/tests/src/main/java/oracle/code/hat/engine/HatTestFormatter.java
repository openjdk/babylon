package oracle.code.hat.engine;

public class HatTestFormatter {

    public static void appendClass(StringBuilder builder, String className) {
        builder.append(Colours.CYAN).append("Class: " + className).append(Colours.RESET).append("\n");;
    }

    public static void testing(StringBuilder builder, String methodName) {
        builder.append(Colours.BLUE)
                .append("Testing: #")
                .append(methodName)
                .append("\t ................... ")
                .append(Colours.RESET);
    }

    public static void ok(StringBuilder builder) {
        builder.append(Colours.GREEN)
                .append("[ok]")
                .append(Colours.RESET)
                .append("\n");;
    }

    public static void fail(StringBuilder builder) {
        builder.append(Colours.RED)
                .append("[fail]")
                .append(Colours.RESET)
                .append("\n");;
    }

}

package jdk.code.tools.renderer;


public final class TerminalColors {
    private TerminalColors() {
    }

    public interface Colorizer {
        String colorize(String text);
    }

    public enum Color implements Colorizer {
        // https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#8-colors
        NONE("0"),
        BLACK("38;5;0"), DARKGREEN("38;5;22"), DARKBLUE("38;5;27"),
        GREY("38;5;247"), RED("38;5;1"), GREEN("38;5;77"), YELLOW("38;5;185"),
        BLUE("38;5;31"), WHITE("38;5;251"), ORANGE("38;5;208"), PURPLE("38;5;133");
        final String escSequence;

        Color(String seq) {
            escSequence = "\u001b[" + seq + "m";
        }

        public String colorize(String string) {
            return (this == NONE) ? string : escSequence + string + NONE.escSequence;
        }
    }
}

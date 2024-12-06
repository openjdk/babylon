package bldr;

import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

record Regex(Pattern pattern) {
    Regex(String regex) {
        this(Pattern.compile(regex));
    }

    public static Regex of(String regexString) {
        return new Regex(regexString);
    }

    boolean matches(String text, Consumer<Matcher> matcherConsumer) {
        if (pattern().matcher(text) instanceof Matcher matcher && matcher.matches()) {
            matcherConsumer.accept(matcher);
            return true;
        } else {
            return false;
        }
    }
}

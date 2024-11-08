import java.lang.runtime.CodeReflection;

public class IR {

    @CodeReflection
    static String add(String a, int b) {
        return a + b;
    }
}

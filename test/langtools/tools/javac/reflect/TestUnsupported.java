/*
 * @test /nodynamiccopyright/
 * @modules jdk.incubator.code
 * @summary Test that unsupported langauge features don't crash javac
 * @compile/fail/ref=TestUnsupported.out -Werror -Xlint:-incubating -XDrawDiagnostics TestUnsupported.java
 */
import jdk.incubator.code.Reflect;

public class TestUnsupported {
    @Reflect
    boolean m(Object o) {
        return switch (o) {
            case String _, Integer _ -> true;
            default -> false;
        };
    }
}

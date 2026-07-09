/*
 * @test /nodynamiccopyright/
 * @modules jdk.incubator.code
 * @ignore there's no more unsupported AST features
 * @summary Test that unsupported langauge features don't crash javac
 * @compile/fail/ref=TestUnsupported.out -Werror -Xlint:-incubating -XDrawDiagnostics TestUnsupported.java
 */
import jdk.incubator.code.Reflect;

public class TestUnsupported {
    @Reflect
    boolean m(Object o) {
        return switch (o) {
            case null, default -> true;
        };
    }
}

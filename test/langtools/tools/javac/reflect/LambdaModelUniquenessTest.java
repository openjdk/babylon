import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit LambdaModelUniquenessTest
 */
public class LambdaModelUniquenessTest {

    Runnable f() {
        return (@CodeReflection Runnable) () -> {
            System.out.println("Running...");
        };
    }

    @Test
    public void testWithLambdaThatDoNotCapture() {
        Runnable q1 = f();
        Runnable q2 = f();
        Quoted quoted1 = Op.ofQuotable(q1).orElseThrow();
        Quoted quoted2 = Op.ofQuotable(q2).orElseThrow();
        Assertions.assertSame(quoted1.op(), quoted2.op());
    }

    @Test
    public void testWithLambdaThatDoNotCapture2() {
        Runnable q1 = f();
        Runnable q2 = f();
        List<Op> ops = Stream.of(q1, q2).parallel().map(q -> Op.ofQuotable(q).orElseThrow().op()).toList();
        Assertions.assertSame(ops.getFirst(), ops.getLast());
    }

    static IntUnaryOperator g(int i) {
        return (@CodeReflection IntUnaryOperator) j -> j + i;
    }

    @Test
    public void testWithLambdaThatCapture() {
        IntUnaryOperator q1 = g(1);
        IntUnaryOperator q2 = g(2);
        Quoted quoted1 = Op.ofQuotable(q1).orElseThrow();
        Quoted quoted2 = Op.ofQuotable(q2).orElseThrow();
        Assertions.assertSame(quoted1.op(), quoted2.op());
    }

    @Test
    public void testWithLambdaThatCapture2() {
        IntUnaryOperator q1 = g(1);
        IntUnaryOperator q2 = g(2);
        List<Op> ops = Stream.of(q1, q2).parallel().map(q -> Op.ofQuotable(q).orElseThrow().op()).toList();
        Assertions.assertSame(ops.getFirst(), ops.getLast());
    }
}

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestOpOfLambda
 */
public class TestOpOfLambda {

    Runnable f() {
        return (@Reflect Runnable) () -> {
            System.out.println("Running...");
        };
    }

    @Test
    public void testWithLambdaThatDoNotCaptureInSequence() {
        Runnable q1 = f();
        Runnable q2 = f();
        Quoted quoted1 = Op.ofLambda(q1).orElseThrow();
        Quoted quoted2 = Op.ofLambda(q2).orElseThrow();
        Assertions.assertSame(quoted1.op(), quoted2.op());
    }

    @Test
    public void testWithLambdaThatDoNotCaptureInParallel() { // parallel
        Runnable q1 = f();
        Runnable q2 = f();
        List<Op> ops = Stream.of(q1, q2).parallel().map(q -> Op.ofLambda(q).orElseThrow().op()).toList();
        Assertions.assertSame(ops.getFirst(), ops.getLast());
    }

    static IntUnaryOperator g(int i) {
        return (@Reflect IntUnaryOperator) j -> j + i;
    }

    @Test
    public void testWithLambdaThatCaptureInSequence() {
        IntUnaryOperator q1 = g(1);
        IntUnaryOperator q2 = g(2);
        Quoted quoted1 = Op.ofLambda(q1).orElseThrow();
        Quoted quoted2 = Op.ofLambda(q2).orElseThrow();
        Assertions.assertSame(quoted1.op(), quoted2.op());
    }

    @Test
    public void testWithLambdaThatCaptureInParallel() {
        IntUnaryOperator q1 = g(1);
        IntUnaryOperator q2 = g(2);
        List<Op> ops = Stream.of(q1, q2).parallel().map(q -> Op.ofLambda(q).orElseThrow().op()).toList();
        Assertions.assertSame(ops.getFirst(), ops.getLast());
    }

    @Test
    public void testQuotedIsSameInSequence() {
        int j = 8;
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i * 2 + j;
        Quoted q1 = Op.ofLambda(q).get();
        Quoted q2 = Op.ofLambda(q).get();
        Assertions.assertSame(q1, q2);
    }

    @Test
    public void testQuotedIsSameInParallel() {
        int j = 8;
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i * 2 + j;
        List<Quoted> quotedObjects = IntStream.range(1, 3).parallel().mapToObj(_ -> Op.ofLambda(q).get()).toList();
        Assertions.assertSame(quotedObjects.getFirst(), quotedObjects.getLast());
    }
}

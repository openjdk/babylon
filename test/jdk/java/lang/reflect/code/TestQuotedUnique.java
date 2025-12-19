import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary For a reflectable lambda, calls to Op.ofQuotable must produce the same instance.
 * @modules jdk.incubator.code
 * @run junit TestQuotedUnique
 */
public class TestQuotedUnique {

    @Test
    public void testInSequence() {
        int j = 8;
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i * 2 + j;
        Quoted q1 = Op.ofQuotable(q).get();
        Quoted q2 = Op.ofQuotable(q).get();
        Assertions.assertSame(q1, q2);
    }

    @Test
    public void testInParallel() {
        int j = 8;
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i * 2 + j;
        List<Quoted> quotedObjects = IntStream.range(1, 3).parallel().mapToObj(_ -> Op.ofQuotable(q).get()).toList();
        Assertions.assertSame(quotedObjects.getFirst(), quotedObjects.getLast());
    }
}

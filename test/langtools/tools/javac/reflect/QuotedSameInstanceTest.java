import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.Quotable;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Op#ofQuotable returns the same instance
 * @modules jdk.incubator.code
 * @run testng QuotedSameInstanceTest
 */

public class QuotedSameInstanceTest {

    private static final Quotable q1 = (Quotable & Runnable) () -> {
    };

    @Test
    void testWithOneThread() {
        Assert.assertSame(Op.ofQuotable(q1).get(), Op.ofQuotable(q1).get());
    }

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }
    private static final QuotableIntUnaryOperator q2 = x -> x;

    @Test
    void testWithMultiThreads() {
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofQuotable(q2).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assert.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }
}

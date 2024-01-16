import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.Quotable;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Quotable#quoted returns the same instance
 * @run testng QuotedSameInstanceTest
 */

public class QuotedSameInstanceTest {

    private static final Quotable q1 = (Quotable & Runnable) () -> {
    };

    @Test
    void testWithOneThread() {
        Assert.assertSame(q1.quoted(), q1.quoted());
    }

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }
    private static final QuotableIntUnaryOperator q2 = x -> x;

    @Test
    void testWithMultiThreads() {
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> q2.quoted()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assert.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }
}

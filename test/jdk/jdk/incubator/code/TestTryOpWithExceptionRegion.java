import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.Inliner;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.IntConsumer;

/*
 * @test
 * @modules jdk.incubator.code
 * @library lib
 * @run junit TestTryOpWithExceptionRegion
 */
public class TestTryOpWithExceptionRegion {
    @Reflect
    static void m(IntConsumer c) {
        try {
            c.accept(0);
            n(c);
        } catch (IllegalArgumentException ex) {
            c.accept(4);
        } finally {
            c.accept(5);
        }
    }

    @Reflect
    private static void n(IntConsumer c) {
        try {
            c.accept(1);
        } catch (IllegalStateException ex) {
            c.accept(2);
        } finally {
            c.accept(3);
        }
    }

    @Test
    void testTryOpEnclosingExceptionRegion() throws NoSuchMethodException {
        // lower n + inline it in m
        CoreOp.FuncOp n = Op.ofMethod(this.getClass().getDeclaredMethod("n", IntConsumer.class)).get();
        CoreOp.FuncOp ln = n.transform(CodeTransformer.LOWERING_TRANSFORMER);
        CoreOp.FuncOp m = Op.ofMethod(this.getClass().getDeclaredMethod("m", IntConsumer.class)).get();
        CoreOp.FuncOp m2 = m.transform((b, o) -> {
            if (o instanceof JavaOp.InvokeOp iop && iop.invokeReference().name().equals("n")) {
                var bb = Inliner.inline(b, ln, b.context().getValues(m.parameters()), (b2, v2) -> {
                });
                return bb.withContextAndTransformer(b.context(), CodeTransformer.COPYING_TRANSFORMER);
            } else {
                b.add(o);
                return b;
            }
        });
        System.out.println(m2.toText());

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), m2, c),
                TestTryOpWithExceptionRegion::m
        );
        test.accept(i -> {
            if (i == 1) throw new IllegalStateException();
        });

        test.accept(i -> {
            if (i == 1) throw new IllegalArgumentException();
        });
    }

    @Test
    void testTryOpEnclosedByExceptionRegion() throws NoSuchMethodException {
        // lower m + inline n
        CoreOp.FuncOp m = Op.ofMethod(this.getClass().getDeclaredMethod("m", IntConsumer.class)).get();
        CoreOp.FuncOp lm = m.transform(CodeTransformer.LOWERING_TRANSFORMER);
        CoreOp.FuncOp n = Op.ofMethod(this.getClass().getDeclaredMethod("n", IntConsumer.class)).get();
        CoreOp.FuncOp lm2 = lm.transform((b, o) -> {
            if (o instanceof JavaOp.InvokeOp iop && iop.invokeReference().name().equals("n")) {
                var bb = Inliner.inline(b, n, b.context().getValues(lm.parameters()), (b2, v2) -> {});
                return bb.withContextAndTransformer(b.context(), CodeTransformer.COPYING_TRANSFORMER);
            } else {
                b.add(o);
                return b;
            }
        });

        Consumer<IntConsumer> test = testConsumer(
                c -> Interpreter.invoke(MethodHandles.lookup(), lm2, c),
                TestTryOpWithExceptionRegion::m
        );
        test.accept(i -> {
            if (i == 1) throw new IllegalStateException();
        });

        test.accept(i -> {
            if (i == 1) throw new IllegalArgumentException();
        });
    }

    static Consumer<IntConsumer> testConsumer(Consumer<IntConsumer> actualR, Consumer<IntConsumer> expectedR) {
        return c -> {
            List<Integer> actual = new ArrayList<>();
            IntConsumer actualC = actual::add;
            Throwable actualT = null;
            try {
                actualR.accept(actualC.andThen(c));
            } catch (Interpreter.InterpreterException e) {
                throw e;
            } catch (Throwable t) {
                actualT = t;
                if (t instanceof AssertionError) {
                    t.printStackTrace();
                }
            }

            List<Integer> expected = new ArrayList<>();
            IntConsumer expectedC = expected::add;
            Throwable expectedT = null;
            try {
                expectedR.accept(expectedC.andThen(c));
            } catch (Throwable t) {
                expectedT = t;
            }

            Assertions.assertEquals(
                    expectedT != null ? expectedT.getClass() : null, actualT != null ? actualT.getClass() : null
            );
            Assertions.assertEquals(expected, actual);
        };
    }
}

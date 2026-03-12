import jdk.incubator.code.*;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestBuildOp
 */
public class TestBuildOp {

    @Reflect
    static List<Integer> f(int i) {
        return new ArrayList<>(i);
    }

    @Test
    void testCopyFromMethod() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp f = Op.ofMethod(m).get();

        assertOpIsCopiedWhenAddedToBlock(f);
    }

    @Test
    void testCopyFromLambda() {
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i / 2;
        Quoted<?> quoted = Op.ofLambda(q).get();
        assert quoted.capturedValues().isEmpty();

        assertOpIsCopiedWhenAddedToBlock(quoted.op());
    }

    @Test
    void testCopyFromOp() {
        CoreOp.ConstantOp constant = CoreOp.constant(JavaType.INT, 7);
        constant.buildAsRoot();

        assertOpIsCopiedWhenAddedToBlock(constant);
    }

    @Test
    void testBuildAsRoot() {
        CoreOp.FuncOp funcOp = CoreOp.func("f", FunctionType.FUNCTION_TYPE_VOID).body(b -> {
            b.op(CoreOp.return_());
        });

        Assertions.assertFalse(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isBound());
        funcOp.buildAsRoot();

        Assertions.assertTrue(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isBound());
        funcOp.buildAsRoot();
    }

    @Test
    void testBuiltLambdaRoot() {
        IntBinaryOperator q = (@Reflect IntBinaryOperator)(int a, int b) -> a + b;
        Quoted<?> quoted = Op.ofLambda(q).orElseThrow();

        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorOp();

        Assertions.assertTrue(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isBound());
    }

    @Test
    void testOpBound() {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        Op.Result r = body.entryBlock().op(CoreOp.constant(JavaType.DOUBLE, 1d));
        body.entryBlock().op(CoreOp.return_());

        Assertions.assertThrows(IllegalStateException.class, () -> r.op().buildAsRoot());
        Assertions.assertTrue(r.op().isBound());
        Assertions.assertFalse(r.op().isRoot());

        CoreOp.func("f", body);
        Assertions.assertTrue(r.op().isBound());
    }

    @Test
    void testSetLocation() {
        CoreOp.ConstantOp cop = CoreOp.constant(JavaType.LONG, 1L);
        cop.setLocation(Op.Location.NO_LOCATION);
        cop.buildAsRoot();

        Assertions.assertThrows(IllegalStateException.class, () -> cop.setLocation(Op.Location.NO_LOCATION));

        IntBinaryOperator q = (@Reflect IntBinaryOperator)(int a, int b) -> a + b;
        Quoted<?> quoted = Op.ofLambda(q).orElseThrow();
        Assertions.assertThrows(IllegalStateException.class, () -> quoted.op().setLocation(Op.Location.NO_LOCATION));
    }

    void assertOpIsCopiedWhenAddedToBlock(Op op) {
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        body.entryBlock().op(op);
        body.entryBlock().op(CoreOp.return_());
        CoreOp.FuncOp funcOp = CoreOp.func("t", body);
        boolean b = funcOp.body().entryBlock().ops().stream().allMatch(o -> o != op);
        Assertions.assertTrue(b);
    }
}

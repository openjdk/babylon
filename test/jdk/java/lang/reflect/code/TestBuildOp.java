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
    void test0() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("f", int.class);
        CoreOp.FuncOp f = Op.ofMethod(m).get();
        assertOpIsCopiedWhenAddedToBlock(f);
    }

    @Test
    void test1() {
        IntUnaryOperator q = (@Reflect IntUnaryOperator) i -> i / 2;
        Quoted<?> quoted = Op.ofLambda(q).get();
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorBody().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().ancestorOp();
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    @Test
    void test2() {
        CoreOp.ConstantOp constant = CoreOp.constant(JavaType.INT, 7);
        constant.buildAsRoot();
        assertOpIsCopiedWhenAddedToBlock(constant);
    }

    @Test
    void test3() {
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
    void test4() {
        IntBinaryOperator q = (@Reflect IntBinaryOperator)(int a, int b) -> {
            return a + b;
        };
        Quoted<?> quoted = Op.ofLambda(q).get();
        CoreOp.QuotedOp quotedOp = (CoreOp.QuotedOp) quoted.op().ancestorBody().ancestorOp();
        CoreOp.FuncOp funcOp = (CoreOp.FuncOp) quotedOp.ancestorBody().ancestorOp();
        Assertions.assertTrue(funcOp.isRoot());
        Assertions.assertFalse(funcOp.isBound());
        assertOpIsCopiedWhenAddedToBlock(funcOp);
    }

    @Test
    void test5() { // freezing an already bound op should throw
        Body.Builder body = Body.Builder.of(null, FunctionType.FUNCTION_TYPE_VOID);
        Op.Result r = body.entryBlock().op(CoreOp.constant(JavaType.DOUBLE, 1d));
        body.entryBlock().op(CoreOp.return_());
        Assertions.assertThrows(IllegalStateException.class, () -> r.op().buildAsRoot());
        Assertions.assertFalse(r.op().isRoot());
        Assertions.assertTrue(r.op().isBound());
        CoreOp.func("f", body);
        Assertions.assertTrue(r.op().isBound());
    }

    @Test
    void test6() {
        CoreOp.ConstantOp cop = CoreOp.constant(JavaType.LONG, 1L);
        cop.setLocation(Op.Location.NO_LOCATION);
        cop.buildAsRoot();
        Assertions.assertThrows(IllegalStateException.class, () -> cop.setLocation(Op.Location.NO_LOCATION));
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

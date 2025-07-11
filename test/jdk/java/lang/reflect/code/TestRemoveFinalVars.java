import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.*;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import static jdk.incubator.code.dialect.core.CoreOp.VarAccessOp.VarLoadOp;
import static jdk.incubator.code.dialect.core.CoreOp.VarAccessOp.VarStoreOp;
import static jdk.incubator.code.dialect.core.CoreOp.VarOp;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestRemoveFinalVars
 */

public class TestRemoveFinalVars {

    @CodeReflection
    static boolean f() {
        final int x = 8; // final var
        int y = x + 2; // final var
        int z = y + 3; // non final var
        z++;
        return x == 8 && y == 10 && z == 14;
    }

    @Test
    void test() {
        FuncOp f = getFuncOp(this.getClass(),"f");
        System.out.println(f.toText());
        FuncOp lf = lower(f);
        System.out.println(lf.toText());

        FuncOp f2 = f.transform(TestRemoveFinalVars::rmFinalVars);
        System.out.println(f2.toText());
        FuncOp lf2 = lower(f2);
        System.out.println(lf2.toText());

        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf), Interpreter.invoke(MethodHandles.lookup(), lf2));

        Op op = SSA.transform(lower(f));
        System.out.println(op.toText());
    }

    static FuncOp lower(FuncOp funcOp) {
        return funcOp.transform(OpTransformer.LOWERING_TRANSFORMER);
    }

    static Block.Builder rmFinalVars(Block.Builder block, Op op) {
        if (op instanceof VarOp varOp) {
            // Is the variable stored to? If not we can remove it
            // otherwise, it's not considered final and we copy it
            if (isValueUsedWithOp(varOp.result(), o -> o instanceof VarStoreOp)) {
                block.op(varOp);
            }
        } else if (op instanceof VarLoadOp varLoadOp) {
            // If the variable is not stored to
            if (!isValueUsedWithOp(varLoadOp.varOp().result(), o -> o instanceof VarStoreOp)) {
                // Map result of load from variable to the value that initialized the variable
                // Subsequently encountered input operations using the result will be copied
                // to output operations using the mapped value
                CopyContext cc = block.context();
                cc.mapValue(varLoadOp.result(), cc.getValue(varLoadOp.varOp().operands().get(0)));
            } else {
                block.op(varLoadOp);
            }
        } else {
            block.op(op);
        }
        return block;
    }

    private static boolean isValueUsedWithOp(Value value, Predicate<Op> opPredicate) {
        for (Op.Result user : value.uses()) {
            if (opPredicate.test(user.op())) {
                return true;
            }
        }
        return false;
    }

    static FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}

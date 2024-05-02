import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOp.FuncOp;
import static java.lang.reflect.code.op.CoreOp.VarAccessOp.VarLoadOp;
import static java.lang.reflect.code.op.CoreOp.VarAccessOp.VarStoreOp;
import static java.lang.reflect.code.op.CoreOp.VarOp;

/*
 * @test
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
        f.writeTo(System.out);
        FuncOp lf = lower(f);
        lf.writeTo(System.out);

        FuncOp f2 = f.transform(TestRemoveFinalVars::rmFinalVars);
        f2.writeTo(System.out);
        FuncOp lf2 = lower(f2);
        lf2.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf), Interpreter.invoke(lf2));

        SSA.transform(lower(f)).writeTo(System.out);
    }

    static FuncOp lower(FuncOp funcOp) {
        return funcOp.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
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
        return m.getCodeModel().get();
    }
}

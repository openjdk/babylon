import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOps.FuncOp;
import static java.lang.reflect.code.op.CoreOps.VarAccessOp.VarLoadOp;
import static java.lang.reflect.code.op.CoreOps.VarAccessOp.VarStoreOp;
import static java.lang.reflect.code.op.CoreOps.VarOp;

/*
 * @test
 * @run testng TestRemoveFinalVars
 */

public class TestRemoveFinalVars {

    @CodeReflection
    static int f(boolean b) {
        final int x = 8; // final var
        int y = x + 2; // final var
        int z = y + 3; // non final var
        z++;
        return y;
    }

    @Test
    void test() {
        FuncOp f = getFuncOp(this.getClass(),"f");
        f.writeTo(System.out);

        FuncOp f2 = f.transform(TestRemoveFinalVars::rmFinalVars);
        f2.writeTo(System.out);
    }

    /*
    if VarOp
        if varOp result not used with VarStore // final variable
            do not copy the varOp
            map varOp result with the initial value // a way to mark the var is final
            do not copy VarLoads on the var and map their result with the init value

    * */

    static Block.Builder rmFinalVars(Block.Builder block, Op op) {
        if (op instanceof VarOp varOp) {
            if (isValueUsedWithOpClass(varOp.result(), VarStoreOp.class)) {
                block.op(varOp);
            }
        } else if (op instanceof VarLoadOp varLoadOp) {
            if (!isValueUsedWithOpClass(varLoadOp.varOp().result(), VarStoreOp.class)) {
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

    private static boolean isValueUsedWithOpClass(Value value, Class<? extends Op> opClass) {
        for (Op.Result user : value.uses()) {
            if (user.op().getClass().equals(opClass)) {
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

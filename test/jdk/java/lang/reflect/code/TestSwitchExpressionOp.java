import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestSwitchExpressionOp
 */
public class TestSwitchExpressionOp {

    @CodeReflection
    public static Object f1(String r) {
        return switch (r) {
            case "FOO" -> "FOO";
            case "BAR" -> "FOO";
            case "BAZ" -> "FOO";
            default -> "";
        };
    }

    @Test
    public void test1() {
        CoreOps.FuncOp f = getFuncOp("f1");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, "FOO"), f1("FOO"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAR"), f1("BAR"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAZ"), f1("BAZ"));
        Assert.assertEquals(Interpreter.invoke(lf, "ELSE"), f1("ELSE"));
    }

    @CodeReflection
    // fallthrough case
    public static Object f2(String r) {
        return switch (r) {
            case "FOO" : {
            }
            case "BAR" : {
                yield "2";
            }
            default : yield "";
        };
    }

    @Test
    public void test2() {
        CoreOps.FuncOp f = getFuncOp("f2");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, "FOO"), f2("FOO"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAR"), f2("BAR"));
        Assert.assertEquals(Interpreter.invoke(lf, "ELSE"), f2("ELSE"));
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSwitchExpressionOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        return om.get().getCodeModel().get();
    }
}

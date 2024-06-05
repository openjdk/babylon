import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestSwitchExpressionOp
 */
public class TestSwitchExpressionOp {

    // TODO more testing
    //  cover cases where MatchException will be thrown

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
        CoreOp.FuncOp lf = lower("f1");

        Assert.assertEquals(Interpreter.invoke(lf, "FOO"), f1("FOO"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAR"), f1("BAR"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAZ"), f1("BAZ"));
        Assert.assertEquals(Interpreter.invoke(lf, "ELSE"), f1("ELSE"));
    }

    @CodeReflection
    public static Object f2(String r) { // switch expr with fallthrough
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
        CoreOp.FuncOp lf = lower("f2");

        Assert.assertEquals(Interpreter.invoke(lf, "FOO"), f2("FOO"));
        Assert.assertEquals(Interpreter.invoke(lf, "BAR"), f2("BAR"));
        Assert.assertEquals(Interpreter.invoke(lf, "ELSE"), f2("ELSE"));
    }

    @CodeReflection
    // null is handled, when selector expr is null the switch will complete normally
    private static String f3(String s) {
        return switch (s) {
            case null -> "null";
            default -> "default";
        };
    }

    @Test
    public void test3() {
        CoreOp.FuncOp lf = lower("f3");

        Assert.assertEquals(Interpreter.invoke(lf, "SOMETHING"), f3("SOMETHING"));
        Assert.assertEquals(Interpreter.invoke(lf, new Object[]{null}), f3(null));
    }

    @CodeReflection
    // null not handled, when selector expr is null it will throw NPE
    private static String f4(String s) {
        return switch (s) {
            default -> "default";
        };
    }

    @Test
    public void test4() {
        CoreOp.FuncOp lf = lower("f4");

        Assert.assertEquals(Interpreter.invoke(lf, "SOMETHING"), f3("SOMETHING"));
        Assert.assertThrows(NullPointerException.class, () -> f4(null));
        Assert.assertThrows(NullPointerException.class, () -> Interpreter.invoke(lf, new Object[]{null}));
    }

    @CodeReflection
    private static String f5(int i) {
        return switch (i) {
            case 1 -> "1";
            case 2 -> "2";
            default -> "default";
        };
    }

    @Test
    public void test5() {
        CoreOp.FuncOp lf = lower("f5");

        Assert.assertEquals(Interpreter.invoke(lf, 1), f5(1));
        Assert.assertEquals(Interpreter.invoke(lf, 2), f5(2));
        Assert.assertEquals(Interpreter.invoke(lf, 99), f5(99));
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSwitchExpressionOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        return om.get().getCodeModel().get();
    }

    private static CoreOp.FuncOp lower(String methodName) {
        return lower(getFuncOp(methodName));
    }

    private static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        return lf;
    }
}

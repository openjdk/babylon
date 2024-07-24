import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
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

    @Test
    void testCaseConstantOtherKindsOfExpr() {
        CoreOp.FuncOp lmodel = lower("caseConstantOtherKindsOfExpr");
        for (int i = 0; i < 14; i++) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, i), caseConstantOtherKindsOfExpr(i));
        }
    }

    static class Constants {
        static final int c1 = 12;
    }

    @CodeReflection
    private static String caseConstantOtherKindsOfExpr(int i) {
        final int eleven = 11;
        return switch (i) {
            case 1 & 0xF -> "1";
            case 4>>1 -> "2";
            case (int) 3L -> "3";
            case 2<<1 -> "4";
            case 10 / 2 -> "5";
            case 12 - 6 -> "6";
            case 3 + 4 -> "7";
            case 2 * 2 * 2 -> "8";
            case 8 | 1 -> "9";
            case (10) -> "10";
            case eleven -> "11";
            case Constants.c1 -> String.valueOf(Constants.c1);
            case 1 > 0 ? 13 : 133 -> "13";
            default -> "an int";
        };
    }

    @Test
    void testCaseConstantEnum() {
        CoreOp.FuncOp lmodel = lower("caseConstantEnum");
        for (Day day : Day.values()) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, day), caseConstantEnum(day));
        }
    }

    enum Day {
        MON, TUE, WED, THU, FRI, SAT, SUN
    }

    @CodeReflection
    private static int caseConstantEnum(Day d) {
        return switch (d) {
            case MON, FRI, SUN -> 6;
            case TUE -> 7;
            case THU, SAT -> 8;
            case WED -> 9;
        };
    }

    @Test
    void testCaseConstantFallThrough() {
        CoreOp.FuncOp lmodel = lower("caseConstantFallThrough");
        char[] args = {'A', 'B', 'C'};
        for (char arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), caseConstantFallThrough(arg));
        }
    }

    @CodeReflection
    private static String caseConstantFallThrough(char c) {
        return switch (c) {
            case 'A':
            case 'B':
                yield "A or B";
            default:
                yield "Neither A nor B";
        };
    }

    // @CodeReflection
    // compiler code doesn't support case null, default
    // @@@ support such as case and test the switch expression lowering for this case
    private static String caseConstantNullAndDefault(String s) {
        return switch (s) {
            case "abc" -> "alphabet";
            case null, default -> "null or default";
        };
    }

    @Test
    void testCaseConstantNullLabel() {
        CoreOp.FuncOp lmodel = lower("caseConstantNullLabel");
        String[] args = {null, "non null"};
        for (String arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), caseConstantNullLabel(arg));
        }
    }

    @CodeReflection
    private static String caseConstantNullLabel(String s) {
        return switch (s) {
            case null -> "null";
            default -> "non null";
        };
    }

    @Test
    void testCaseConstantThrow() {
        CoreOp.FuncOp lmodel = lower("caseConstantThrow");
        Assert.assertThrows(IllegalArgumentException.class, () -> Interpreter.invoke(lmodel, 8));
        int[] args = {9, 10};
        for (int arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), caseConstantThrow(arg));
        }
    }

    @CodeReflection
    private static String caseConstantThrow(Integer i) {
        return switch (i) {
            case 8 -> throw new IllegalArgumentException();
            case 9 -> "NINE";
            default -> "An integer";
        };
    }

    @Test
    void testCaseConstantMultiLabels() {
        CoreOp.FuncOp lmodel = lower("caseConstantMultiLabels");
        char[] args = {'a', 'e', 'i', 'o', 'u', 'j', 'p', 'g'};
        for (char arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), caseConstantMultiLabels(arg));
        }
    }

    @CodeReflection
    private static String caseConstantMultiLabels(char c) {
        return switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u': yield "vowel";
            default: yield "consonant";
        };
    }

    @Test
    void testCaseConstantBehaviorIsSyntaxIndependent() {
        CoreOp.FuncOp ruleExpression = lower("caseConstantRuleExpression");
        CoreOp.FuncOp ruleBlock = lower("caseConstantRuleBlock");
        CoreOp.FuncOp statement = lower("caseConstantStatement");

        String[] args = {"FOO", "BAR", "BAZ", "OTHER"};

        for (String arg : args) {
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(ruleBlock, arg));
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(statement, arg));
        }
    }

    @CodeReflection
    public static String caseConstantRuleExpression(String r) {
        return switch (r) {
            case "FOO" -> "BAR";
            case "BAR" -> "BAZ";
            case "BAZ" -> "FOO";
            default -> "";
        };
    }

    @CodeReflection
    public static String caseConstantRuleBlock(String r) {
        return switch (r) {
            case "FOO" -> {
                yield "BAR";
            }
            case "BAR" -> {
                yield "BAZ";
            }
            case "BAZ" -> {
                yield "FOO";
            }
            default -> {
                yield "";
            }
        };
    }

    @CodeReflection
    private static String caseConstantStatement(String s) {
        return switch (s) {
            case "FOO": yield "BAR";
            case "BAR": yield "BAZ";
            case "BAZ": yield "FOO";
            default: yield "";
        };
    }

    private static CoreOp.FuncOp lower(String methodName) {
        return lower(getCodeModel(methodName));
    }

    private static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        return lf;
    }

    private static CoreOp.FuncOp getCodeModel(String methodName) {
        Optional<Method> om = Stream.of(TestSwitchExpressionOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();

        return om.get().getCodeModel().get();
    }
}

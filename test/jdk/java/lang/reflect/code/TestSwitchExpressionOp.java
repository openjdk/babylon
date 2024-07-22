import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestSwitchExpressionOp
 */
public class TestSwitchExpressionOp {

    @Test
    void testCasePatternGuard() {
        CoreOp.FuncOp lmodel = lower("casePatternGuard");
        Object[] args = {"c++", "java", new R(8), new R(2L), new R(3f), new R(4.0)};
        for (Object arg : args) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, arg), casePatternGuard(arg));
        }
    }
    @CodeReflection
    static String casePatternGuard(Object obj) {
        return switch (obj) {
            case String s when s.length() > 3 -> "str with length > %d".formatted(s.length());
            case R(Number n) when n.getClass().equals(Double.class) -> "R(Double)";
            default -> "else";
        };
    }

    @Test
    void testCaseRecordPattern() {
        // @@@ new R(null) must match the pattern R(Number c), but it doesn't
        // @@@ test with generic record
        CoreOp.FuncOp lmodel = lower("caseRecordPattern");
        Object[] args = {new R(8), new R(1.0), new R(2L), "abc"};
        for (Object arg : args) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, arg), caseRecordPattern(arg));
        }
    }
    record R(Number n) {}
    @CodeReflection
    static String caseRecordPattern(Object o) {
        return switch (o) {
            case R(Number _) -> "R(_)";
            default -> "else";
        };
    }
    @Test
    void testCaseTypePattern() {
        CoreOp.FuncOp lmodel = lower("caseTypePattern");
        Object[] args = {"str", new ArrayList<>(), new int[]{}, new Stack[][]{}, new Collection[][][]{}, 8, 'x'};
        for (Object arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), caseTypePattern(arg));
        }
    }
    @CodeReflection
    static String caseTypePattern(Object o) {
        return switch (o) {
            case String _ -> "String"; // class
            case RandomAccess _ -> "RandomAccess"; // interface
            case int[] _ -> "int[]"; // array primitive
            case Stack[][] _ -> "Stack[][]"; // array class
            case Collection[][][] _ -> "Collection[][][]"; // array interface
            case final Number n -> "Number"; // final modifier
            default -> "something else";
        };
    }

    @Test
    void testCasePatternWithCaseConstant() {
        CoreOp.FuncOp lmodel = lower("casePatternWithCaseConstant");
        int[] args = {42, 43, -44, 0};
        for (int arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), casePatternWithCaseConstant(arg));
        }
    }

    @CodeReflection
    static String casePatternWithCaseConstant(Integer a) {
        return switch (a) {
            case 42 -> "forty two";
            // @@@ case int will not match, because of the way InstanceOfOp is interpreted
            case Integer i when i > 0 -> "positive int";
            case Integer i when i < 0 -> "negative int";
            default -> "zero";
        };
    }

    // @Test
    void testCasePatternMultiLabel() {
        CoreOp.FuncOp lmodel = lower("casePatternMultiLabel");
        Object[] args = {(byte) 1, (short) 2, 'A', 3, 4L, 5f, 6d, true, "str"};
        for (Object arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), casePatternMultiLabel(arg));
        }
    }
    // @CodeReflection
    // code model for such as code is not supported
    // @@@ support this case and uncomment its test
    private static String casePatternMultiLabel(Object o) {
        return switch (o) {
            case Integer _, Long _, Character _, Byte _, Short _-> "integral type";
            default -> "non integral type";
        };
    }

    @Test
    void testCasePatternThrow() {
        CoreOp.FuncOp lmodel = lower("casePatternThrow");

        Object[] args = {Byte.MAX_VALUE, Short.MIN_VALUE, 0, 1L, 11f, 22d};
        for (Object arg : args) {
            Assert.assertThrows(IllegalArgumentException.class, () -> Interpreter.invoke(lmodel, arg));
        }

        Object[] args2 = {"abc", List.of()};
        for (Object arg : args2) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), casePatternThrow(arg));
        }
    }

    @CodeReflection
    private static String casePatternThrow(Object o) {
        return switch (o) {
            case Number n -> throw new IllegalArgumentException();
            case String s -> "a string";
            default -> o.getClass().getName();
        };
    }

    @Test
    void testCasePatternBehaviorIsSyntaxIndependent() {
        CoreOp.FuncOp ruleExpression = lower("casePatternRuleExpression");
        CoreOp.FuncOp ruleBlock = lower("casePatternRuleBlock");
        CoreOp.FuncOp statement = lower("casePatternStatement");

        String[] args = {"FOO", "BAR", "BAZ", "OTHER"};

        for (String arg : args) {
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(ruleBlock, arg));
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(statement, arg));
        }
    }

    @CodeReflection
    private static String casePatternRuleExpression(Object o) {
        return switch (o) {
            case Integer i -> "integer";
            case String s -> "string";
            default -> "not integer nor string";
        };
    }

    @CodeReflection
    private static String casePatternRuleBlock(Object o) {
        return switch (o) {
            case Integer i -> {
                yield "integer";
            }
            case String s -> {
                yield "string";
            }
            default -> {
                yield "not integer nor string";
            }
        };
    }

    @CodeReflection
    private static String casePatternStatement(Object o) {
        return switch (o) {
            case Integer i: yield "integer";
            case String s: yield "string";
            default: yield "not integer nor string";
        };
    }

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
            case 0xF | 1 -> "9";
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

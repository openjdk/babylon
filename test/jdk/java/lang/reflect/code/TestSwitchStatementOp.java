import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.writer.OpWriter;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
* @test
* @run testng TestSwitchStatementOp
* */
public class TestSwitchStatementOp {

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
        String s = "";
        switch (r) {
            case "FOO" -> s += "BAR";
            case "BAR" -> s += "BAZ";
            case "BAZ" -> s += "FOO";
            default -> s += "else";
        }
        return s;
    }

    @CodeReflection
    public static String caseConstantRuleBlock(String r) {
        String s = "";
        switch (r) {
            case "FOO" -> {
                s += "BAR";
            }
            case "BAR" -> {
                s += "BAZ";
            }
            case "BAZ" -> {
                s += "FOO";
            }
            default -> {
                s += "else";
            }
        }
        return s;
    }

    @CodeReflection
    private static String caseConstantStatement(String s) {
        String r = "";
        switch (s) {
            case "FOO":
                r += "BAR";
                break;
            case "BAR":
                r += "BAZ";
                break;
            case "BAZ":
                r += "FOO";
                break;
            default:
                r += "else";
        }
        return r;
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
        String r = "";
        switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u':
                r += "vowel";
                break;
            default:
                r += "consonant";
        }
        return r;
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
        String r = "";
        switch (i) {
            case 8 -> throw new IllegalArgumentException();
            case 9 -> r += "Nine";
            default -> r += "An integer";
        }
        return r;
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
        String r = "";
        switch (s) {
            case null -> r += "null";
            default -> r += "non null";
        }
        return r;
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
        String r = "";
        switch (c) {
            case 'A':
            case 'B':
                r += "A or B";
                break;
            default:
                r += "Neither A nor B";
        }
        return r;
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
    private static String caseConstantEnum(Day d) {
        String r = "";
        switch (d) {
            case MON, FRI, SUN -> r += 6;
            case TUE -> r += 7;
            case THU, SAT -> r += 8;
            case WED -> r += 9;
        }
        return r;
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
        String r = "";
        final int eleven = 11;
        switch (i) {
            case 1 & 0xF -> r += 1;
            case 4>>1 -> r += "2";
            case (int) 3L -> r += 3;
            case 2<<1 -> r += 4;
            case 10 / 2 -> r += 5;
            case 12 - 6 -> r += 6;
            case 3 + 4 -> r += 7;
            case 2 * 2 * 2 -> r += 8;
            case 8 | 1 -> r += 9;
            case (10) -> r += 10;
            case eleven -> r += 11;
            case Constants.c1 -> r += Constants.c1;
            case 1 > 0 ? 13 : 133 -> r += 13;
            default -> r += "an int";
        }
        return r;
    }

    @Test
    void testCaseConstantConv() {
        CoreOp.FuncOp lmodel = lower("caseConstantConv");
        for (short i = 1; i < 5; i++) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, i), caseConstantConv(i));
        }
    }

    @CodeReflection
    static String caseConstantConv(short a) {
        final short s = 1;
        final byte b = 2;
        String r = "";
        switch (a) {
            case s -> r += "one"; // identity, short -> short
            case b -> r += "two"; // widening primitive conversion, byte -> short
            case 3 -> r += "three"; // narrowing primitive conversion, int -> short
            default -> r += "else";
        }
        return r;
    }

    @Test
    void testCaseConstantConv2() {
        CoreOp.FuncOp lmodel = lower("caseConstantConv2");
        Byte[] args = {1, 2, 3};
        for (Byte arg : args) {
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lmodel, arg), caseConstantConv2(arg));
        }
    }

    @CodeReflection
    static String caseConstantConv2(Byte a) {
        final byte b = 2;
        String r = "";
        switch (a) {
            case 1 -> r+= "one"; // narrowing primitive conversion followed by a boxing conversion, int -> bye -> Byte
            case b -> r+= "two"; // boxing, byte -> Byte
            default -> r+= "default";
        }
        return r;
    }

    private static CoreOp.FuncOp lower(String methodName) {
        return lower(getCodeModel(methodName));
    }

    private static CoreOp.FuncOp lower(CoreOp.FuncOp f) {
        writeModel(f, System.out, OpWriter.LocationOption.DROP_LOCATION);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        writeModel(lf, System.out, OpWriter.LocationOption.DROP_LOCATION);

        return lf;
    }

    private static void writeModel(CoreOp.FuncOp f, OutputStream os, OpWriter.Option... options) {
        StringWriter sw = new StringWriter();
        new OpWriter(sw, options).writeOp(f);
        try {
            os.write(sw.toString().getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static CoreOp.FuncOp getCodeModel(String methodName) {
        Optional<Method> om = Stream.of(TestSwitchStatementOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();

        return om.get().getCodeModel().get();
    }
}

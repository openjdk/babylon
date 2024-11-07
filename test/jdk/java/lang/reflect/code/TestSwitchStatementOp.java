import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.writer.OpWriter;
import jdk.incubator.code.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

/*
* @test
* @modules jdk.incubator.code
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

    @Test
    void testNonEnhancedSwStatNoDefault() {
        CoreOp.FuncOp lmodel = lower("nonEnhancedSwStatNoDefault");
        for (int i = 1; i < 4; i++) {
            Assert.assertEquals(Interpreter.invoke(lmodel, i), nonEnhancedSwStatNoDefault(i));
        }
    }

    @CodeReflection
    static String nonEnhancedSwStatNoDefault(int a) {
        String r = "";
        switch (a) {
            case 1 -> r += "1";
            case 2 -> r += 2;
        }
        return r;
    }

    // no reason to test enhanced switch statement that has no default
    // because we can't test for MatchException without separate compilation

    @Test
    void testEnhancedSwStatUnconditionalPattern() {
        CoreOp.FuncOp lmodel = lower("enhancedSwStatUnconditionalPattern");
        String[] args = {"A", "B"};
        for (String arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), enhancedSwStatUnconditionalPattern(arg));
        }
    }

    @CodeReflection
    static String enhancedSwStatUnconditionalPattern(String s) {
        String r = "";
        switch (s) {
            case "A" -> r += "A";
            case Object o -> r += "obj";
        }
        return r;
    }

    @Test
    void testCasePatternBehaviorIsSyntaxIndependent() {
        CoreOp.FuncOp ruleExpression = lower("casePatternRuleExpression");
        CoreOp.FuncOp ruleBlock = lower("casePatternRuleBlock");
        CoreOp.FuncOp statement = lower("casePatternStatement");

        Object[] args = {1, "2", 3L};

        for (Object arg : args) {
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(ruleBlock, arg));
            Assert.assertEquals(Interpreter.invoke(ruleExpression, arg), Interpreter.invoke(statement, arg));
        }
    }

    @CodeReflection
    private static String casePatternRuleExpression(Object o) {
        String r = "";
        switch (o) {
            case Integer i -> r += "integer";
            case String s -> r+= "string";
            default -> r+= "else";
        }
        return r;
    }

    @CodeReflection
    private static String casePatternRuleBlock(Object o) {
        String r = "";
        switch (o) {
            case Integer i -> {
                r += "integer";
            }
            case String s -> {
                r += "string";
            }
            default -> {
                r += "else";
            }
        }
        return r;
    }

    @CodeReflection
    private static String casePatternStatement(Object o) {
        String r = "";
        switch (o) {
            case Integer i:
                r += "integer";
                break;
            case String s:
                r += "string";
                break;
            default:
                r += "else";
        }
        return r;
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
        String r = "";
        switch (o) {
            case Number n -> throw new IllegalArgumentException();
            case String s -> r += "a string";
            default -> r += o.getClass().getName();
        }
        return r;
    }

    // @@@ when multi patterns is supported, we will test it

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
        String r = "";
        switch (a) {
            case 42 -> r += "forty two";
            // @@@ case int will not match, because of the way InstanceOfOp is interpreted
            case Integer i when i > 0 -> r += "positive int";
            case Integer i when i < 0 -> r += "negative int";
            default -> r += "zero";
        }
        return r;
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
        String r = "";
        switch (o) {
            case String _ -> r+= "String"; // class
            case RandomAccess _ -> r+= "RandomAccess"; // interface
            case int[] _ -> r+= "int[]"; // array primitive
            case Stack[][] _ -> r+= "Stack[][]"; // array class
            case Collection[][][] _ -> r+= "Collection[][][]"; // array interface
            case final Number n -> r+= "Number"; // final modifier
            default -> r+= "something else";
        }
        return r;
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
        String r = "";
        switch (o) {
            case R(Number n) -> r += "R(_)";
            default -> r+= "else";
        }
        return r;
    }

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
        String r = "";
        switch (obj) {
            case String s when s.length() > 3 -> r += "str with length > %d".formatted(s.length());
            case R(Number n) when n.getClass().equals(Double.class) -> r += "R(Double)";
            default -> r += "else";
        }
        return r;
    }

    @Test
    void testDefaultCaseNotTheLast() {
        CoreOp.FuncOp lmodel = lower("defaultCaseNotTheLast");
        String[] args = {"something", "M", "A"};
        for (String arg : args) {
            Assert.assertEquals(Interpreter.invoke(lmodel, arg), defaultCaseNotTheLast(arg));
        }
    }

    @CodeReflection
    static String defaultCaseNotTheLast(String s) {
        String r = "";
        switch (s) {
            default -> r += "else";
            case "M" -> r += "Mow";
            case "A" -> r += "Aow";
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

        return CoreOp.ofMethod(om.get()).get();
    }
}

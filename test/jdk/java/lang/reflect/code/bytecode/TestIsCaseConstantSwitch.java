import jdk.incubator.code.*;
import jdk.incubator.code.bytecode.impl.LoweringTransform;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.java.JavaOp.*;

/*
 * @test
 * @modules jdk.incubator.code/jdk.incubator.code.bytecode.impl
 * @enablePreview
 * @compile TestIsCaseConstantSwitch.java
 * @run junit TestIsCaseConstantSwitch
 */
public class TestIsCaseConstantSwitch {

    class C {
        static final int x = 26;
    }

    @Reflect
    private static void caseConstantSwitchExpressions() {
        // switch label
        // case label
        // list of case constant
        // every case constant must be either a constant expression or the name of an enum constant
        // null literal
        // list of case patterns
        // default label
        final int fv = 25;
        int i = -1;
        String r = switch (i) {
            // literal of primitive type
            case 1 -> "A";
            // unary operators +, -, ~
            case +2 -> "B";
            case -2 -> "BB";
            case ~2 -> "BBB"; // -3
            // multiplicative operators *, /, %
            case 3 * 4 -> "E";
            case 3 / 4 -> "EE";
            case 3 % 4 -> "EEE";
            // shift operators <<, >>, >>>
            case 4 << 5 -> "F"; // 128
            case 10 >> 1 -> "FF"; // 5
            case 8 >>> 1 -> "FFF"; // 4
            // relational operators <, <=, >, >= (and conditional operator)
            case 1 < 2 ? 9 : 10 -> "G"; // 9
            case 1 <= 2 ? 11 : 12 -> "GG"; // 11
            case 1 > 2 ? 13 : 14 -> "GGG"; // 14
            case 1 >= 2 ? 15 : 16 -> "GGGG"; // 16
            // equality operators ==, !=
            case 1 == 2 ? 17 : 18 -> "H"; // 18
            case 1 != 2 ? 19 : 20 -> "HH"; // 19
            // bitwise and logical operators &, ^, |
            case 6 & 6 -> "I"; // 6
            case 7 ^ 8 -> "II"; // 15
            case 8 | 10 -> "III"; // 10
            // conditional-and operator &&
            case 2 > 3 && 5 > 6 ? 21 : 22 -> "J"; // 22
            case 2 > 3 || 5 > 6 ? 23 : 24 -> "JJ"; // 24
            // parenthesized expressions whose contained expression is a constant expression
            case (20) -> "K";
            // simple names that refer to constant variables
            case fv -> "L";
            // qualified names of the form TypeName.Identifier that refer to constant variables
            case C.x -> "M";
            // list of case constants
            case 21, 30 -> null;
            // casts
            case (int) 31L -> "N";
            case (int) 34f -> "NN";
            // default
            default -> "X";
        };

        // we can have a target of type Byte, Short, Character, Integer
        // as long as we don't introduce case null, javac will generate labels identical to what we have in source code
        Integer ii = -2;
        r = switch (ii) {
            case 1 -> "A";
            default -> "X";
        };

        char c = '2';
        r = switch (c) {
            case '1' -> "1";
            default -> "";
        };
    }

    enum E {
        V;
    }

    @Reflect
    static void nonCaseConstantSwitchExpressions() {
        int r;

        String s = "";
        r = switch (s) {
            case "A" -> 1;
            default -> 0;
        };

        E e = E.V;
        r = switch (e) {
            case V -> 1;
        };

        boolean b = false;
        r = switch (b) {
            case true -> 1;
            default -> 0;
        };

        long l = 5L;
        r = switch (l) {
            case 1L -> 1;
            default -> 0;
        };

        float f = 5f;
        r = switch (f) {
            case 1f -> 1;
            default -> 0;
        };

        double d = 5d;
        r = switch (d) {
            case 1d -> 1;
            default -> 0;
        };

        Integer i = 4;
        r = switch (i) {
            case 1 -> 1;
            case null -> -1;
            default -> 0;
        };
    }

    static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of("caseConstantSwitchExpressions", true),
                Arguments.of("nonCaseConstantSwitchExpressions", false)
        );
    }

    @ParameterizedTest
    @MethodSource("cases")
    void testIsConstantLabelSwitch(String methodName, boolean expected) throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod(methodName);
        CoreOp.FuncOp codeModel = Op.ofMethod(m).get();
        List<SwitchExpressionOp> swExprOps = codeModel.body().entryBlock().ops().stream()
                .filter(o -> o instanceof SwitchExpressionOp)
                .map(o -> ((SwitchExpressionOp) o)).toList();
        for (SwitchExpressionOp swExprOp : swExprOps) {
            Assertions.assertEquals(
                    new LoweringTransform.ConstantLabelSwitchChecker(swExprOp, MethodHandles.lookup()).isCaseConstantSwitch(),
                    expected,
                    swExprOp.toText());
        }
    }

    @Test
    void testGettingLabels() throws NoSuchMethodException {
        var expectedLabels = List.of(1, +2, -2, ~2, 12, 3 / 4, 3 % 4, 4 << 5, 10 >> 1,
                8 >>> 1, 1 < 2 ? 9 : 10, 1 <= 2 ? 11 : 12, 1 > 2 ? 13 : 14, 1 >= 2 ? 15 : 16, 1 == 2 ? 17 : 18,
                1 != 2 ? 19 : 20, 6 & 6, 7 ^ 8, 8 | 10, 2 > 3 && 5 > 6 ? 21 : 22, 2 > 3 || 5 > 6 ? 23 : 24, (20), 25,
                C.x, 21, 30, (int) 31L, (int) 34f );
        var funcOp = Op.ofMethod(this.getClass().getDeclaredMethod("caseConstantSwitchExpressions")).get();
        System.out.println(funcOp.toText());
        var swOp = (JavaSwitchOp) funcOp.body().entryBlock().ops().stream().filter(op -> op instanceof JavaSwitchOp).findFirst().get();
        List<Integer> actualLabels = getLabelsAndTargets(MethodHandles.lookup(), swOp);
        System.out.println(actualLabels);
        Assertions.assertEquals(expectedLabels, actualLabels);
    }

    static ArrayList<Integer> getLabelsAndTargets(MethodHandles.Lookup lookup, JavaSwitchOp swOp) {
        var labels = new ArrayList<Integer>();
        for (int i = 0; i < swOp.bodies().size(); i += 2) {
            Body labelBody = swOp.bodies().get(i);
            labels.addAll(getLabels(lookup, labelBody));
        }
        return labels;
    }

    static List<Integer> getLabels(MethodHandles.Lookup lookup, Body body) {
        if (body.blocks().size() != 1 || !(body.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop) ||
                !(yop.yieldValue() instanceof Result opr)) {
            throw new IllegalStateException("Body of a java switch fails the expected structure");
        }
        var labels = new ArrayList<Integer>();
        if (opr.op() instanceof EqOp eqOp) {
            labels.add(extractConstantLabel(lookup, body, eqOp));
        } else if (opr.op() instanceof InvokeOp invokeOp &&
                invokeOp.invokeReference().equals(MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class))) {
            labels.add(extractConstantLabel(lookup, body, invokeOp));
        } else if (opr.op() instanceof ConditionalOrOp cor) {
            for (Body corbody : cor.bodies()) {
                labels.addAll(getLabels(lookup, corbody));
            }
        } else if (!(opr.op() instanceof CoreOp.ConstantOp)){ // not default label
            throw new IllegalStateException();
        }
        return labels;
    }

    static Integer extractConstantLabel(MethodHandles.Lookup lookup, Body body, Op whenToStop) {
        Op lastOp = body.entryBlock().ops().get(body.entryBlock().ops().indexOf(whenToStop) - 1);
        CoreOp.FuncOp funcOp = CoreOp.func("f", CoreType.functionType(lastOp.result().type())).body(block -> {
            // in case we refer to constant variables in the label
            for (Value capturedValue : body.capturedValues()) {
                if (!(capturedValue instanceof Result r) || !(r.op() instanceof CoreOp.VarOp vop)) {
                    continue;
                }
                block.op(((Result) vop.initOperand()).op());
                block.op(vop);
            }
            Result last = null;
            for (Op op : body.entryBlock().ops()) {
                if (op.equals(whenToStop)) {
                    break;
                }
                last = block.op(op);
            }
            block.op(CoreOp.return_(last));
        });
        Object res = Interpreter.invoke(lookup, funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER));
        return switch (res) {
            case Byte b -> Integer.valueOf(b);
            case Short s -> Integer.valueOf(s);
            case Character c -> Integer.valueOf(c);
            case Integer i -> i;
            default -> throw new IllegalStateException(); // @@@ not going to happen
        };
    }
}

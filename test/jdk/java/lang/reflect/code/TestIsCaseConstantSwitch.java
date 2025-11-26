import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.java.JavaOp.*;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestIsCaseConstantSwitch
 */
public class TestIsCaseConstantSwitch {

    @Test
    void testIsConstantLabelSwitch() throws NoSuchMethodException {
        Method m = this.getClass().getDeclaredMethod("caseConstantSwitchExpressions", int.class, E.class);
        CoreOp.FuncOp codeModel = Op.ofMethod(m).get();
        List<SwitchExpressionOp> swExprOps = codeModel.body().entryBlock().ops().stream()
                .filter(o -> o instanceof SwitchExpressionOp)
                .map(o -> ((SwitchExpressionOp) o)).toList();
        for (SwitchExpressionOp swExprOp : swExprOps) {
            Assertions.assertTrue(swExprOp.isCaseConstantSwitch(MethodHandles.lookup()), swExprOp.toText());
        }

        // test with methods in TestSwitchExpressionOp and in TestSwitchStatementOp
        List<CoreOp.FuncOp> funcOps = Stream.concat(
                        Arrays.stream(TestSwitchExpressionOp.class.getDeclaredMethods()),
                        Arrays.stream(TestSwitchStatementOp.class.getDeclaredMethods()))
                .filter(dm -> dm.getName().startsWith("caseConstant") && dm.isAnnotationPresent(Reflect.class))
                .map(dm -> Op.ofMethod(dm).get())
                .toList();
        for (CoreOp.FuncOp fop : funcOps) {
            JavaSwitchOp swOp = ((JavaSwitchOp) fop.body().entryBlock().ops().stream().filter(o -> o instanceof JavaSwitchOp).findFirst().get());
            Assertions.assertTrue(swOp.isCaseConstantSwitch(MethodHandles.lookup()), fop.toText());
        }
    }

    class C {
        static final int x = 26;
    }

    enum E {
        V;
    }

    @Reflect
    private static void caseConstantSwitchExpressions(int i, E e) {
        // switch label
            // case label
                // list of case constant
                    // every case constant must be either a constant expression or the name of an enum constant
                // null literal
                // list of case patterns
            // default label
        final int fv = 25;
        String r = switch (i) {
            // literal of primitive type
            case 1 -> "A";
            // unary operators +, -, ~
            case +2 -> "B";
            case -2 -> "BB";
            case ~2 -> "BBB";
            // multiplicative operators *, /, %
            case 3 * 4 -> "E";
            case 3 / 4 -> "EE";
            case 3 % 4 -> "EEE";
            // shift operators <<, >>, >>>
            case 4 << 5 -> "F";
            case 10 >> 1 -> "FF";
            case 8 >>> 1 -> "FFF";
            // relational operators <, <=, >, >= (and conditional operator)
            case 1 < 2 ? 9 : 10 -> "G";
            case 1 <= 2 ? 11 : 12 -> "GG";
            case 1 > 2 ? 13 : 14 -> "GGG";
            case 1 >= 2 ? 15 : 16 -> "GGGG";
            // equality operators ==, !=
            case 1 == 2 ? 17 : 18 -> "H";
            case 1 != 2 ? 19 : 20 -> "HH";
            // bitwise and logical operators &, ^, |
            case 6 & 6 -> "I";
            case 7 ^ 8 -> "II";
            case 8 | 10 -> "III";
            // conditional-and operator &&
            case 2 > 3 && 5 > 6 ? 21 : 22 -> "J";
            case 2 > 3 || 5 > 6 ? 23 : 24 -> "JJ";
            // parenthesized expressions whose contained expression is a constant expression
            case (20) -> "K";
            // simple names that refer to constant variables
            case fv -> "L";
            // qualified names of the form TypeName.Identifier that refer to constant variables
             case C.x -> "M";
             // list of case constants
            case 21, 30 -> null;
            default -> "X";
        };

        String r2 = switch (r) {
            // string literal
            case "A" -> "A+";
            case null -> "A++";
            default -> "X";
        };

        String r3 = switch (e) {
            // name of an enum constant
             case E.V -> "V";
            case null -> "N";
            default -> "D";
        };

    }

}

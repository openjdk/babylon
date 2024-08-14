import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
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

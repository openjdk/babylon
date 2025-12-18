import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpWriter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.StringWriter;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Set;
import java.util.function.IntBinaryOperator;

/*
 * @test
 * @modules jdk.incubator.code
 * @modules java.base/java.lang.invoke:open
 * @library ../
 * @run junit TestTransform
 * @run main Unreflect TestTransform
 * @run junit TestTransform
 */
public class TestTransform {

    @Reflect
    static int f() {
        IntBinaryOperator o = (a, b) -> a + b;
        int sum = 0;
        for (int i = 0; i < 10; i++) {
            sum += o.applyAsInt(i, i);
        }
        return sum + 42;
    }

    static int add(int a, int b) {
        return a + b;
    }

    @Reflect
    static int fWithAddMethod() {
        IntBinaryOperator o = (a, b) -> add(a, b);
        int sum = 0;
        for (int i = 0; i < 10; i = add(i, 1)) {
            sum = add(sum, o.applyAsInt(i, i));
        }
        return add(sum, 42);
    }

    @Test
    public void testOpTransformer_fWithAddMethod() throws Exception {
        Method fAdd = this.getClass().getDeclaredMethod("add", int.class, int.class);
        var codeTransformer = CodeTransformer.opTransformer((builder, op, operands) -> {
            switch (op) {
                case JavaOp.AddOp _ ->
                        builder.apply(JavaOp.invoke(MethodRef.method(fAdd), operands));
                default ->
                        builder.apply(op);
            }
        });

        testTransformer("f", "fWithAddMethod", codeTransformer);
    }

    @Test
    public void testCodeTransformer_fWithAddMethod() throws Exception {
        Method fAdd = this.getClass().getDeclaredMethod("add", int.class, int.class);
        CodeTransformer codeTransformer = (builder, op) -> {
            switch (op) {
                case JavaOp.AddOp _ -> {
                    List<Value> values = builder.context().getValues(op.operands());
                    Op.Result r = builder.op(JavaOp.invoke(MethodRef.method(fAdd), values));
                    builder.context().mapValue(op.result(), r);
                }
                default ->
                        builder.op(op);
            }
            return builder;
        };

        testTransformer("f", "fWithAddMethod", codeTransformer);
    }


    @Reflect
    static int fAddToSubNeg() {
        IntBinaryOperator o = (a, b) -> a - -b;
        int sum = 0;
        for (int i = 0; i < 10; i = i - -1) {
            sum = sum - -o.applyAsInt(i, i);
        }
        return sum - -42;
    }

    @Test
    public void testOpTransformer_fAddToSubNeg() throws Exception {
        var codeTransformer = CodeTransformer.opTransformer((builder, op, operands) -> {
            switch (op) {
                case CoreOp.ConstantOp _ -> {
                    Set<Op.Result> uses = op.result().uses();
                    // add(x, constant(C))
                    if (uses.size() == 1 && uses.iterator().next().op() instanceof JavaOp.AddOp) {
                        // Drop, this will be replaced later
                    } else {
                        builder.apply(op);
                    }
                }
                case JavaOp.AddOp _ -> {
                    // add(x, constant(C)) -> sub(x, constant(-C))
                    // add(x, y) -> sub(x, neg(y))
                    Op.Result rhs;
                    if (op.operands().get(1) instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp cop) {
                        // There is no mapping to the second operand, since it was associated
                        // with the constant op which was dropped
                        Assertions.assertNull(operands.get(1));

                        rhs = builder.apply(CoreOp.constant(JavaType.INT, -(int) cop.value()));
                    } else {
                        Assertions.assertNotNull(operands.get(1));

                        rhs = builder.apply(JavaOp.neg(operands.get(1)));
                    }
                    builder.apply(JavaOp.sub(operands.get(0), rhs));
                }
                default ->
                        builder.apply(op);
            }
        });

        testTransformer("f", "fAddToSubNeg", codeTransformer);
    }

    @Test
    public void testCodeTransformer_fAddToSubNeg() throws Exception {
        CodeTransformer codeTransformer = (builder, op) -> {
            switch (op) {
                case CoreOp.ConstantOp _ -> {
                    Set<Op.Result> uses = op.result().uses();
                    // add(x, constant(C))
                    if (uses.size() == 1 && uses.iterator().next().op() instanceof JavaOp.AddOp) {
                        // Drop, this will be replaced later
                    } else {
                        builder.op(op);
                    }
                }
                case JavaOp.AddOp _ -> {
                    // add(x, constant(C)) -> sub(x, constant(-C))
                    // add(x, y) -> sub(x, neg(y))
                    List<Value> operands = op.operands().stream()
                            .map(v -> builder.context().getValueOrDefault(v, null)).toList();
                    Op.Result rhs;
                    if (op.operands().get(1) instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp cop) {
                        // There is no mapping to the second operand, since it was associated
                        // with the constant op which was dropped
                        Assertions.assertNull(operands.get(1));

                        rhs = builder.op(CoreOp.constant(JavaType.INT, - (int) cop.value()));
                    } else {
                        Assertions.assertNotNull(operands.get(1));

                        rhs = builder.op(JavaOp.neg(operands.get(1)));
                    }
                    Op.Result result = builder.op(JavaOp.sub(operands.get(0), rhs));
                    builder.context().mapValue(op.result(), result);
                }
                default ->
                        builder.op(op);
            }
            return builder;
        };

        testTransformer("f", "fAddToSubNeg", codeTransformer);
    }


     void testTransformer(String methodName, String transformedMethodName, CodeTransformer codeTransformer) throws Exception {
        Method fMethod = this.getClass().getDeclaredMethod(methodName);
        var fModel = Op.ofMethod(fMethod).orElseThrow();

        var fTransformed = fModel.transform(codeTransformer);

        Method fTransformedMethod = this.getClass().getDeclaredMethod(transformedMethodName);
        var fTransformedModel = Op.ofMethod(fTransformedMethod).orElseThrow();

        assertEqual(fTransformedModel, fTransformed, methodName, transformedMethodName);
    }

    static void assertEqual(Op expected, Op actual,
                            String methodName, String transformedMethodName) {
        Assertions.assertEquals(serialize(expected).replace(transformedMethodName, methodName), serialize(actual));
    }

    static String serialize(Op o) {
        StringWriter w = new StringWriter();
        OpWriter.writeTo(w, o, OpWriter.LocationOption.DROP_LOCATION);
        return w.toString();
    }

}

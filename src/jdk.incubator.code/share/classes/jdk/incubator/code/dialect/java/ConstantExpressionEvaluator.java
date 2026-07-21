package jdk.incubator.code.dialect.java;

import jdk.incubator.code.Body;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.internal.ArithmeticAndConvOpImpls;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import static jdk.incubator.code.dialect.java.JavaType.J_L_STRING;
import static jdk.incubator.code.dialect.java.JavaType.VOID;

final class ConstantExpressionEvaluator {
    private final MethodHandles.Lookup l;
    private final Map<Value, Object> m = new HashMap<>();

    ConstantExpressionEvaluator(MethodHandles.Lookup l) {
        this.l = l;
    }

    <T extends Op & JavaOp.JavaExpression> Optional<Object> evaluate(T op) {
        try {
            Object v = this.eval(op);
            return Optional.ofNullable(v);
        } catch (ArithmeticAndConvOpImpls.NonConstantExpression e) {
            return Optional.empty();
        }
    }

    Optional<Object> evaluate(Value v) {
        try {
            Object o = this.eval(v);
            return Optional.ofNullable(o);
        } catch (ArithmeticAndConvOpImpls.NonConstantExpression e) {
            return Optional.empty();
        }
    }

    private Object eval(Op op) {
        if (m.containsKey(op.result())) {
            return m.get(op.result());
        }
        Object r = switch (op) {
            case CoreOp.ConstantOp cop when isConstant(cop) -> {
                Object v = cop.value();
                yield v instanceof String s ? s.intern() : v;
            }
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp when varLoadOp.operands().getFirst() instanceof Op.Result &&
                    isConstant(varLoadOp.varOp()) -> eval(varLoadOp.varOp().initOperand());
            case JavaOp.ConvOp _ -> {
                // we expect cast to primitive type
                var v = eval(op.operands().getFirst());
                yield ArithmeticAndConvOpImpls.evaluate(op, List.of(v));
            }
            case JavaOp.CastOp castOp -> {
                // we expect cast to String
                Value operand = castOp.operands().getFirst();
                if (!castOp.resultType().equals(J_L_STRING) || !operand.type().equals(J_L_STRING)) {
                    throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                }
                Object v = eval(operand);
                if (!(v instanceof String s)) {
                    throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                }
                yield s;
            }
            case JavaOp.ConcatOp concatOp -> {
                Object first = eval(concatOp.operands().getFirst());
                Object second = eval(concatOp.operands().getLast());
                yield (first.toString() + second).intern();
            }
            case JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp -> {
                Field field;
                VarHandle vh;
                try {
                    field = fieldLoadOp.fieldReference().resolveToField(l);
                    vh = fieldLoadOp.fieldReference().resolveToHandle(l);
                } catch (ReflectiveOperationException | IllegalArgumentException _) {
                    // we cann't reflectivelly get the field
                    throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                }
                // Requirement: the field must be a constant variable.
                // Current checks:
                // 1) The field is declared final.
                // 2) The field type is a primitive or String.
                // Missing check:
                // 3) Verify the field is initialized and the initializer is a constant expression.
                if ((field.getModifiers() & Modifier.FINAL) == 0 ||
                        !isConstantType(fieldLoadOp.fieldReference().type())) {
                    throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                }
                if ((field.getModifiers() & Modifier.STATIC) != 0) {
                    Object v;
                    try {
                        v = vh.get();
                    } catch (Throwable t) {
                        throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                    }
                    if (!isConstantValue(v)) {
                        throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                    }
                    yield v instanceof String s ? s.intern() : v;
                } else {
                    // we can't get the value of an instance field from the model
                    // we need the value of the receiver
                    throw new ArithmeticAndConvOpImpls.NonConstantExpression();
                }
            }
            case JavaOp.ArithmeticOperation _ -> {
                List<Object> values = op.operands().stream().map(this::eval).toList();
                yield ArithmeticAndConvOpImpls.evaluate(op, values);
            }
            case JavaOp.ConditionalExpressionOp _ -> {
                boolean p = evalBoolean(op.bodies().get(0));
                Object t = eval(op.bodies().get(1));
                Object f = eval(op.bodies().get(2));
                yield p ? t : f;
            }
            case JavaOp.ConditionalAndOp _ -> {
                boolean left = evalBoolean(op.bodies().get(0));
                boolean right = evalBoolean(op.bodies().get(1));
                yield left && right;
            }
            case JavaOp.ConditionalOrOp _ -> {
                boolean left = evalBoolean(op.bodies().get(0));
                boolean right = evalBoolean(op.bodies().get(1));
                yield left || right;
            }
            default -> throw new ArithmeticAndConvOpImpls.NonConstantExpression();
        };
        m.put(op.result(), r);
        return r;
    }

    private Object eval(Value v) {
        if (v.declaringElement() instanceof JavaOp.JavaExpression e) {
            return eval((Op & JavaOp.JavaExpression) e);
        }
        throw new ArithmeticAndConvOpImpls.NonConstantExpression();
    }

    private Object eval(Body body) throws ArithmeticAndConvOpImpls.NonConstantExpression {
        if (body.blocks().size() != 1 ||
                !(body.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop) ||
                yop.yieldValue() == null ||
                !isConstantType(yop.yieldValue().type())) {
            throw new ArithmeticAndConvOpImpls.NonConstantExpression();
        }
        return eval(yop.yieldValue());
    }

    private boolean evalBoolean(Body body) throws ArithmeticAndConvOpImpls.NonConstantExpression {
        Object eval = eval(body);
        if (!(eval instanceof Boolean b)) {
            throw new ArithmeticAndConvOpImpls.NonConstantExpression();
        }
        return b;
    }

    private static boolean isConstant(CoreOp.ConstantOp op) {
        return isConstantType(op.resultType()) && isConstantValue(op.value());
    }

    private static boolean isConstant(CoreOp.VarOp op) {
        // Requirement: the local variable must be a constant variable.
        // Current checks:
        // 1) The variable is initialized, and the initializer is a constant expression.
        // 2) The variable type is a primitive or String.
        // Missing check:
        // 3) Ensure the variable is declared final
        return isConstantType(op.varValueType()) &&
                !op.isUninitialized() &&
                // @@@ Add to VarOp
                op.result().uses().stream().noneMatch(u -> u.op() instanceof CoreOp.VarAccessOp.VarStoreOp);
    }

    private static boolean isConstantValue(Object o) {
        return switch (o) {
            case String _ -> true;
            case Boolean _, Byte _, Short _, Character _, Integer _, Long _, Float _, Double _ -> true;
            case null, default -> false;
        };
    }

    private static boolean isConstantType(CodeType e) {
        return (e instanceof PrimitiveType && !VOID.equals(e)) || J_L_STRING.equals(e);
    }
}

package experiments;


import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.function.Predicate;

public class TestQuoted {
    public class TestLambdaMethodRef {

        interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable {
        }

        interface QuotableFunction<T, R> extends Function<T, R>, Quotable {
        }

        interface QuotableBiFunction<T, U, R> extends BiFunction<T, U, R>, Quotable {
        }

        // @Test
        public void test() {
        /*
        lambda (%0 : int)int -> {
            %1 : Var<int> = var %0 @"x$0";
            %2 : int = var.load %1;
            %3 : int = invoke %2 @"TestLambda::m1(int)int";
            return %3;
        };
         */
            QuotableIntUnaryOperator f1 = TestLambdaMethodRef::m1;
            isMethodRef(f1);

        /*
        lambda (%0 : int)int -> {
            %1 : Var<int> = var %0 @"x$0";
            %2 : int = var.load %1;
            %3 : java.lang.Integer = invoke %2 @"java.lang.Integer::valueOf(int)java.lang.Integer";
            %4 : java.lang.Integer = invoke %3 @"TestLambda::m2(java.lang.Integer)java.lang.Integer";
            %5 : int = invoke %4 @"java.lang.Integer::intValue()int";
            return %5;
        };
         */
            QuotableIntUnaryOperator f2 = TestLambdaMethodRef::m2;
            isMethodRef(f2);

        /*
        lambda (%0 : java.lang.Integer)java.lang.Integer -> {
            %1 : Var<java.lang.Integer> = var %0 @"x$0";
            %2 : java.lang.Integer = var.load %1;
            %3 : int = invoke %2 @"java.lang.Integer::intValue()int";
            %4 : int = invoke %3 @"TestLambda::m1(int)int";
            %5 : java.lang.Integer = invoke %4 @"java.lang.Integer::valueOf(int)java.lang.Integer";
            return %5;
        };
         */
            QuotableFunction<Integer, Integer> f3 = TestLambdaMethodRef::m1;
            isMethodRef(f3);

        /*
        lambda (%0 : java.lang.Integer)java.lang.Integer -> {
            %1 : Var<java.lang.Integer> = var %0 @"x$0";
            %2 : java.lang.Integer = var.load %1;
            %3 : java.lang.Integer = invoke %2 @"TestLambda::m2(java.lang.Integer)java.lang.Integer";
            return %3;
        };
         */
            QuotableFunction<Integer, Integer> f4 = TestLambdaMethodRef::m2;
            isMethodRef(f4);

        /*
        lambda (%0 : int)int -> {
            %1 : Var<int> = var %0 @"x$0";
            %2 : int = var.load %1;
            %3 : int = invoke %4 %2 @"TestLambda::m3(int)int";
            return %3;
        };
         */
            QuotableIntUnaryOperator f5 = this::m3;
            isMethodRef(f5);

        /*
        lambda (%0 : TestLambda, %1 : java.lang.Integer)java.lang.Integer -> {
            %2 : Var<TestLambda> = var %0 @"x$0";
            %3 : Var<java.lang.Integer> = var %1 @"x$1";
            %4 : TestLambda = var.load %2;
            %5 : java.lang.Integer = var.load %3;
            %6 : int = invoke %5 @"java.lang.Integer::intValue()int";
            %7 : int = invoke %4 %6 @"TestLambda::m4(TestLambda, int)int";
            %8 : java.lang.Integer = invoke %7 @"java.lang.Integer::valueOf(int)java.lang.Integer";
            return %8;
        };
         */
            QuotableBiFunction<TestLambdaMethodRef, Integer, Integer> f6 = TestLambdaMethodRef::m4;
            isMethodRef(f6);
        }

        static void isMethodRef(Quotable q) {
            Quoted quoted = q.quoted();
            CoreOp.LambdaOp op = (CoreOp.LambdaOp) quoted.op();
            System.out.println(isMethodRef(op));
        }

        static boolean isMethodRef(CoreOp.LambdaOp lambdaOp) {
            // Single block
            if (lambdaOp.body().blocks().size() > 1) {
                return false;
            }

            // zero or one (this) capture
            List<Value> cvs = lambdaOp.capturedValues();
            if (cvs.size() > 1) {
                return false;
            }

            Map<Value, Value> valueMapping = new HashMap<>();
            CoreOp.InvokeOp methodRefInvokeOp = extractMethodInvoke(valueMapping, lambdaOp.body().entryBlock().ops());
            if (methodRefInvokeOp == null) {
                return false;
            }

            // Lambda's parameters map in encounter order with method invocations operands
            List<Value> lambdaParameters = new ArrayList<>();
            if (cvs.size() == 1) {
                lambdaParameters.add(cvs.getFirst());
            }
            lambdaParameters.addAll(lambdaOp.parameters());
            List<Value> methodRefOperands = methodRefInvokeOp.operands().stream().map(valueMapping::get).toList();
            return lambdaParameters.equals(methodRefOperands);
        }

        static CoreOp.InvokeOp extractMethodInvoke(Map<Value, Value> valueMapping, List<Op> ops) {
            CoreOp.InvokeOp methodRefInvokeOp = null;
            for (Op op : ops) {
                switch (op) {
                    case CoreOp.VarOp varOp -> {
                        if (isValueUsedWithOp(varOp.result(), o -> o instanceof CoreOp.VarAccessOp.VarStoreOp)) {
                            return null;
                        }
                    }
                    case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                        Value v = varLoadOp.varOp().operands().getFirst();
                        valueMapping.put(varLoadOp.result(), valueMapping.getOrDefault(v, v));
                    }
                    case CoreOp.InvokeOp iop when isBoxOrUnboxInvocation(iop) -> {
                        Value v = iop.operands().getFirst();
                        valueMapping.put(iop.result(), valueMapping.getOrDefault(v, v));
                    }
                    case CoreOp.InvokeOp iop -> {
                        if (methodRefInvokeOp != null) {
                            return null;
                        }

                        for (Value o : iop.operands()) {
                            valueMapping.put(o, valueMapping.getOrDefault(o, o));
                        }
                        methodRefInvokeOp = iop;
                    }
                    case CoreOp.ReturnOp rop -> {
                        if (methodRefInvokeOp == null) {
                            return null;
                        }
                        Value r = rop.returnValue();
                        if (!(valueMapping.getOrDefault(r, r) instanceof Op.Result invokeResult)) {
                            return null;
                        }
                        if (invokeResult.op() != methodRefInvokeOp) {
                            return null;
                        }
                        assert methodRefInvokeOp.result().uses().size() == 1;
                    }
                    default -> {
                        return null;
                    }
                }
            }

            return methodRefInvokeOp;
        }

        static final Set<String> UNBOX_NAMES = Set.of(
                "byteValue",
                "shortValue",
                "charValue",
                "intValue",
                "longValue",
                "floatValue",
                "doubleValue",
                "booleanValue");

    //    static final Collection<TypeElement> BOX_TYPES = JavaType.primitiveToWrapper.values();

        private static boolean isBoxOrUnboxInvocation(CoreOp.InvokeOp iop) {
            MethodRef mr = iop.invokeDescriptor();
            return false;// BOX_TYPES.contains(mr.refType()) && (UNBOX_NAMES.contains(mr.name()) || mr.name().equals("valueOf"));
        }

        private static boolean isValueUsedWithOp(Value value, Predicate<Op> opPredicate) {
            for (Op.Result user : value.uses()) {
                if (opPredicate.test(user.op())) {
                    return true;
                }
            }
            return false;
        }

        static int m1(int i) {
            return i;
        }

        static Integer m2(Integer i) {
            return i;
        }

        int m3(int i) {
            return i;
        }

        static int m4(TestLambdaMethodRef tl, int i) {
            return i;
        }
    }
}

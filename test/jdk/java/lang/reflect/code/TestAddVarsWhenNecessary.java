import jdk.incubator.code.*;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.*;
import org.junit.Test;

import java.io.PrintStream;
import java.lang.invoke.MethodType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestAddVarsWhenNecessary
 */
public class TestAddVarsWhenNecessary {

    static final MethodRef MAP_PUT = MethodRef.method(Map.class, "put",
            MethodType.methodType(Object.class, Object.class, Object.class));
    static final JavaType MAP = JavaType.type(Map.class.describeConstable().get());
    static final JavaType HASH_MAP = JavaType.type(HashMap.class.describeConstable().get());
    static CoreOp.FuncOp f() {
        Body.Builder body = Body.Builder.of(null, FunctionType.functionType(MAP));
        Block.Builder block = body.entryBlock();
        Op.Result map = block.op(CoreOp._new(
                JavaType.parameterized(MAP, JavaType.J_L_INTEGER, JavaType.J_L_INTEGER),
                FunctionType.functionType(HASH_MAP)));
        Op.Result c1 = block.op(CoreOp.constant(JavaType.INT, 1));
        Op.Result c2 = block.op(CoreOp.constant(JavaType.INT, 2));
        block.op(CoreOp.invoke(MAP_PUT, map, c1, c2));
        block.op(CoreOp._return(map));
        return CoreOp.func("f", body);
    }
    static CoreOp.FuncOp g() {
        Body.Builder body = Body.Builder.of(null, FunctionType.functionType(JavaType.INT));
        Block.Builder block = body.entryBlock();
        Op.Result var = block.op(CoreOp.var(block.op(CoreOp.constant(JavaType.INT, 1))));
        block.op(CoreOp.varStore(var, block.op(CoreOp.constant(JavaType.INT, 2))));
        block.op(CoreOp._return(block.op(CoreOp.varLoad(var))));
        return CoreOp.func("g", body);
    }

    static final MethodRef PRINT_INT = MethodRef.method(PrintStream.class, "print",
            MethodType.methodType(Void.class, int.class));

    static CoreOp.FuncOp h() {
        Body.Builder body = Body.Builder.of(null, FunctionType.functionType(JavaType.INT, JavaType.INT));
        Block.Builder block = body.entryBlock();
        Block.Parameter p = block.parameters().get(0);
        // @@@ do we need the type to construct a FieldRef ??
        Op.Result sout = block.op(CoreOp.fieldLoad(FieldRef.field(System.class, "out", PrintStream.class)));
        block.op(CoreOp.invoke(PRINT_INT, sout, p));
        block.op(CoreOp._return(p));
        return CoreOp.func("h", body);
    }

    @Test
    public void test() {
        CoreOp.FuncOp f = f();
        f.writeTo(System.out);

        f = addVarsWhenNecessary(f);
        f.writeTo(System.out);
    }

    @Test
    public void test2() {
        CoreOp.FuncOp g = g();
        g.writeTo(System.out);

        g = addVarsWhenNecessary(g);
        g.writeTo(System.out);
    }

    @Test
    public void test3() {
        CoreOp.FuncOp h = h();
        h.writeTo(System.out);

        h = addVarsWhenNecessary(h);
        h.writeTo(System.out);
    }

    public static CoreOp.FuncOp addVarsWhenNecessary(CoreOp.FuncOp funcOp) {
        // using cc only is not possible
        // because at first opr --> varOpRes
        // at the first usage we would have to opr --> varLoad
        // meaning we would have to back up the mapping, update it, then restore it before transforming the next op

        Map<Value, CoreOp.VarOp> valueToVar = new HashMap<>();

        return CoreOp.func(funcOp.funcName(), funcOp.body().bodyType()).body(block -> {
            var newParams = block.parameters();
            var oldParams = funcOp.parameters();
            for (int i = 0; i < newParams.size(); i++) {
                Op.Result var = block.op(CoreOp.var(newParams.get(i)));
                valueToVar.put(oldParams.get(i), ((CoreOp.VarOp) var.op()));
            }

            block.transformBody(funcOp.body(), List.of(), (Block.Builder b, Op op) -> {
                var cc = b.context();
                for (Value operand : op.operands()) {
                    if (valueToVar.containsKey(operand)) {
                        var varLoadRes = b.op(CoreOp.varLoad(valueToVar.get(operand).result()));
                        cc.mapValue(operand, varLoadRes);
                    }
                }
                var opr = b.op(op);
                if (!(op instanceof CoreOp.VarOp) && op.result().uses().size() > 1) {
                    var varOpRes = b.op(CoreOp.var(opr));
                    valueToVar.put(op.result(), ((CoreOp.VarOp) varOpRes.op()));
                }
                return b;
            });
        });
    }

}

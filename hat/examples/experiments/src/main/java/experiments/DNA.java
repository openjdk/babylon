
package experiments;

import java.lang.reflect.Method;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;

public class DNA {
    static int myFunc(int i) {
        return 0;
    }

    @CodeReflection
    public static void addMul(int add, int mul) {
        int len = myFunc(add);
    }

    public static class DNAOp extends Op { // externalized
        private final TypeElement type;

        DNAOp(String opName, TypeElement type, List<Value> operands) {
            super(opName, operands);
            this.type = type;
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            throw new IllegalStateException("in transform");
            //  return null;
        }


        @Override
        public TypeElement resultType() {
            System.out.println("in result type");
            return type;
        }
    }


    static public void main(String[] args) throws Exception {
        Method method = DNA.class.getDeclaredMethod("addMul", int.class, int.class);
        var funcOp = method.getCodeModel().get();
        var transformed = funcOp.transform((builder, op) -> {
            CopyContext cc = builder.context();
            if (op instanceof CoreOp.InvokeOp invokeOp) {
               // List<Value> operands = new ArrayList<>();
                //builder.op(new DNAOp("dna", JavaType.INT, operands));
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = cc.getValues(inputOperands);
                Op.Result inputResult = invokeOp.result();
                Op.Result outputResult = builder.op(new DNAOp("dna", JavaType.INT, outputOperands));
                cc.mapValue(inputResult, outputResult);
            } else {
                builder.op(op);
            }
            return builder;
        });


        System.out.println(transformed.toText());

    }
}


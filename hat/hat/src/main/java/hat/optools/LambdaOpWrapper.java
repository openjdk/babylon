package hat.optools;

import hat.util.Result;

import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.MethodRef;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LambdaOpWrapper extends OpWrapper<CoreOp.LambdaOp> {
    public LambdaOpWrapper(CoreOp.LambdaOp op) {
        super(op);
    }

    public InvokeOpWrapper getInvoke(int index) {
        var result = new Result<CoreOp.InvokeOp>();
        selectOnlyBlockOfOnlyBody(blockWrapper ->
                result.of(blockWrapper.op(index))
        );
        return OpWrapper.wrap(result.get());
    }

    public List<Value> operands() {
        return op().operands();
    }

    public Method getQuotableTargetMethod() {
        return getQuotableTargetInvokeOpWrapper().method();
    }

    public InvokeOpWrapper getQuotableTargetInvokeOpWrapper() {
        return OpWrapper.wrap(op().body().entryBlock().ops().stream()
                .filter(op -> op instanceof CoreOp.InvokeOp)
                .map(op -> (CoreOp.InvokeOp) op)
                .findFirst().get());
    }

    public MethodRef getQuotableTargetMethodRef() {
        return getQuotableTargetInvokeOpWrapper().methodRef();
    }

    public Object[] getQuotableCapturedValues(Quoted quoted, Method method) {
        var block = op().body().entryBlock();
        var ops = block.ops();
        Object[] varLoadNames = ops.stream()
                .filter(op -> op instanceof CoreOp.VarAccessOp.VarLoadOp)
                .map(op -> (CoreOp.VarAccessOp.VarLoadOp) op)
                .map(varLoadOp -> (Op.Result) varLoadOp.operands().get(0))
                .map(varLoadOp -> (CoreOp.VarOp) varLoadOp.op())
                .map(varOp -> varOp.varName()).toArray();


        Map<String, Object> nameValueMap = new HashMap<>();

        quoted.capturedValues().forEach((k, v) -> {
            if (k instanceof Op.Result result) {
                if (result.op() instanceof CoreOp.VarOp varOp) {
                    nameValueMap.put(varOp.varName(), v);
                }
            }
        });
        Object[] args = new Object[method.getParameterCount()];
        if (args.length != varLoadNames.length) {
            throw new IllegalStateException("Why don't we have enough captures.!! ");
        }
        for (int i = 1; i < args.length; i++) {
            args[i] = nameValueMap.get(varLoadNames[i].toString());
            if (args[i] instanceof CoreOp.Var varbox) {
                args[i] = varbox.value();
            }
        }
        return args;
    }
}

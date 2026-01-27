package hat.phases;

import hat.callgraph.KernelCallGraph;
import hat.dialect.HATMathLibOp;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record HATMathLibPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {

        Map<Op, ReducedFloatType> setTypeMap = new HashMap<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(invoke -> !invoke.returnsVoid() && HATPhaseUtils.isMathLib(invoke))
                .forEach(invoke ->
                        invoke.op().result().uses().stream()
                            .filter(result -> (result.op() instanceof CoreOp.VarOp) || (result.op() instanceof CoreOp.VarAccessOp.VarStoreOp))
                            .findFirst()
                            .ifPresent(result -> {
                                ReducedFloatType reducedFloatType =  HATFP16Phase.categorizeReducedFloatFromResult(invoke.op());
                                setTypeMap.put(result.op(), reducedFloatType);
                                setTypeMap.put(invoke.op(), reducedFloatType);
                            }));

        // Note: to allow composition of ops, we need another phase in which we process the store to another invokeOp/mathFunction.
        //       In this case we could create F16.of (in the case of F16) for the intermediate results.

        return Trxfmr.of(this, funcOp).transform(setTypeMap::containsKey, (blockBuilder, op) -> {
            switch (op) {
                case JavaOp.InvokeOp invokeOp -> {
                    // Invoke Op is replaced with a HATMathLibOp
                    List<Value> operands = blockBuilder.context().getValues(invokeOp.operands());

                    // For each operand, obtain if it is a reference from global memory or device memory:
                    List<Boolean> referenceList = IntStream.range(0, operands.size())
                            .mapToObj(i -> HATPhaseUtils.isArrayReference(lookup(), invokeOp.operands().get(i)))
                            .collect(Collectors.toList());

                    HATMathLibOp hatMathLibOp = new HATMathLibOp(
                            invokeOp.resultType(),
                            invokeOp.invokeDescriptor().name(),  // intrinsic name
                            setTypeMap.get(invokeOp),
                            referenceList,
                            operands);

                    Op.Result hatMathLibOpResult = blockBuilder.op(hatMathLibOp);
                    blockBuilder.context().mapValue(invokeOp.result(), hatMathLibOpResult);
                }

                case CoreOp.VarOp varOp -> {
                    if (setTypeMap.get(varOp) == null) {
                        // this means that the varOp is not a special type
                        // then we insert the varOp into the new tree
                        blockBuilder.op(varOp);
                    } else {
                        // Add the special type as a VarOp
                        HATFP16Phase.createF16VarOp(varOp, blockBuilder, setTypeMap.get(varOp));
                    }
                }
                default -> // We might need to process stores
                        blockBuilder.op(op);
            }
            return blockBuilder;
        }).funcOp();
    }
}

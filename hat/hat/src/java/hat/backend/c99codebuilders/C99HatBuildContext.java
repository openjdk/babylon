package hat.backend.c99codebuilders;

import hat.optools.ForOpWrapper;
import hat.optools.FuncOpWrapper;
import hat.optools.IfOpWrapper;
import hat.optools.OpWrapper;
import hat.optools.WhileOpWrapper;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

public class C99HatBuildContext {


    static class Scope<OW extends OpWrapper<?>> {
        final Scope<?> parent;
        final OW opWrapper;

        public Scope(Scope<?> parent, OW opWrapper) {
            this.parent = parent;
            this.opWrapper = opWrapper;
        }

        public  CoreOp.VarOp resolve(Value value){
            if (value instanceof  Op.Result result && result.op() instanceof CoreOp.VarOp varOp){
                return varOp;
            }
            if (parent != null){
                return parent.resolve(value);
            }
            throw new IllegalStateException("failed to resolve VarOp for value "+value);
        }
    }

    static class FuncScope extends Scope<FuncOpWrapper> {
        FuncScope(Scope<?> parent, FuncOpWrapper funcOpWrapper) {
            super(parent, funcOpWrapper);
        }

        @Override
        public CoreOp.VarOp resolve(Value value) {
            if (value instanceof Block.Parameter blockParameter) {
                if (opWrapper.parameterToVarOpMap.containsKey(blockParameter)) {
                    return  opWrapper.parameterToVarOpMap.get(blockParameter);
                } else {
                    throw new IllegalStateException("what ?");
                }
            } else {
                return super.resolve(value);
            }
        }
    }

    static abstract class LoopScope<T extends OpWrapper<?>> extends Scope<T>{

        public LoopScope(Scope<?> parent, T opWrapper) {
            super(parent, opWrapper);
        }
    }



    static class ForScope extends LoopScope<ForOpWrapper> {
        Map<Block.Parameter, CoreOp.VarOp> blockParamToVarOpMap = new HashMap<>();

        ForOpWrapper forOpWrapper() {
            return opWrapper;
        }

        ForScope(Scope<?> parent, ForOpWrapper forOpWrapper) {
            super(parent, forOpWrapper);
            var loopParams = forOpWrapper().op().loopBody().blocks().getFirst().parameters().toArray(new Block.Parameter[0]);
            var updateParams = forOpWrapper().op().update().blocks().getFirst().parameters().toArray(new Block.Parameter[0]);
            var condParams = forOpWrapper().op().cond().blocks().getFirst().parameters().toArray(new Block.Parameter[0]);
            var lastInitOp = forOpWrapper().op().init().blocks().getFirst().ops().getLast();
            var lastInitOpOperand0Result = (Op.Result) lastInitOp.operands().getFirst();
            var lastInitOpOperand0ResultOp = lastInitOpOperand0Result.op();
            CoreOp.VarOp varOps[];
            if (lastInitOpOperand0ResultOp instanceof CoreOp.TupleOp tupleOp) {
                 /*
                 for (int j = 1, i=2, k=3; j < size; k+=1,i+=2,j+=3) {
                    float sum = k+i+j;
                 }
                 java.for
                 ()Tuple<Var<int>, Var<int>, Var<int>> -> {
                     %0 : int = constant @"1";
                     %1 : Var<int> = var %0 @"j";
                     %2 : int = constant @"2";
                     %3 : Var<int> = var %2 @"i";
                     %4 : int = constant @"3";
                     %5 : Var<int> = var %4 @"k";
                     %6 : Tuple<Var<int>, Var<int>, Var<int>> = tuple %1 %3 %5;
                     yield %6;
                 }
                 (%7 : Var<int>, %8 : Var<int>, %9 : Var<int>)boolean -> {
                     %10 : int = var.load %7;
                     %11 : int = var.load %12;
                     %13 : boolean = lt %10 %11;
                     yield %13;
                 }
                 (%14 : Var<int>, %15 : Var<int>, %16 : Var<int>)void -> {
                     %17 : int = var.load %16;
                     %18 : int = constant @"1";
                     %19 : int = add %17 %18;
                     var.store %16 %19;
                     %20 : int = var.load %15;
                     %21 : int = constant @"2";
                     %22 : int = add %20 %21;
                     var.store %15 %22;
                     %23 : int = var.load %14;
                     %24 : int = constant @"3";
                     %25 : int = add %23 %24;
                     var.store %14 %25;
                     yield;
                 }
                 (%26 : Var<int>, %27 : Var<int>, %28 : Var<int>)void -> {
                     %29 : int = var.load %28;
                     %30 : int = var.load %27;
                     %31 : int = add %29 %30;
                     %32 : int = var.load %26;
                     %33 : int = add %31 %32;
                     %34 : float = conv %33;
                     %35 : Var<float> = var %34 @"sum";
                     java.continue;
                 };
                 */
                varOps = tupleOp.operands().stream().map(operand -> (CoreOp.VarOp) (((Op.Result) operand).op())).toList().toArray(new CoreOp.VarOp[0]);
            } else {
                 /*
                 for (int j = 0; j < size; j+=1) {
                    float sum = j;
                 }
                 java.for
                    ()Var<int> -> {
                        %0 : int = constant @"0";
                        %1 : Var<int> = var %0 @"j";
                        yield %1;
                    }
                    (%2 : Var<int>)boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = var.load %5;
                        %6 : boolean = lt %3 %4;
                        yield %6;
                    }
                    (%7 : Var<int>)void -> {
                        %8 : int = var.load %7;
                        %9 : int = constant @"1";
                        %10 : int = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<int>)void -> {
                        %12 : int = var.load %11;
                        %13 : float = conv %12;
                        %14 : Var<float> = var %13 @"sum";
                        java.continue;
                    };

                 */
                varOps = new CoreOp.VarOp[]{(CoreOp.VarOp) lastInitOpOperand0ResultOp};
            }
            for (int i = 0; i < varOps.length; i++) {
                blockParamToVarOpMap.put(condParams[i], varOps[i]);
                blockParamToVarOpMap.put(updateParams[i], varOps[i]);
                blockParamToVarOpMap.put(loopParams[i], varOps[i]);
            }
        }


        @Override
        public CoreOp.VarOp resolve(Value value) {
            if (value instanceof Block.Parameter blockParameter) {
                CoreOp.VarOp varOp = this.blockParamToVarOpMap.get(blockParameter);
                if (varOp != null){
                    return varOp;
                }
            }
            return super.resolve(value);
        }
    }

    static class IfScope extends Scope<IfOpWrapper> {
        IfScope(Scope<?> parent, IfOpWrapper opWrapper) {
            super(parent, opWrapper);
        }
    }
    static class WhileScope extends LoopScope<WhileOpWrapper> {
        WhileScope(Scope<?> parent, WhileOpWrapper opWrapper) {
            super(parent, opWrapper);
        }

    }
    Scope<?> scope = null;

    private void popScope() {
        scope = scope.parent;
    }

    private void pushScope(OpWrapper<?> opWrapper) {
        scope = switch (opWrapper) {
            case FuncOpWrapper $ -> new FuncScope(scope, $);
            case ForOpWrapper $ -> new ForScope(scope, $);
            case IfOpWrapper $ -> new IfScope(scope, $);
            case WhileOpWrapper $ -> new WhileScope(scope, $);
            default -> new Scope<>(scope, opWrapper);
        };
    }
    public void scope(OpWrapper<?> opWrapper, Runnable r) {
        pushScope(opWrapper);
        r.run();
        popScope();
    }
    FuncOpWrapper funcOpWrapper;

    C99HatBuildContext(FuncOpWrapper funcOpWrapper) {
        this.funcOpWrapper = funcOpWrapper;
    }

}

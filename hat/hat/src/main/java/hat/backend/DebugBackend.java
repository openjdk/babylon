package hat.backend;

import hat.ComputeContext;
import hat.OpsAndTypes;
import hat.NDRange;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;

public class DebugBackend extends BackendAdaptor {
    public enum HowToRunCompute{REFLECT, BABYLON_INTERPRETER, BABYLON_CLASSFILE}
    public HowToRunCompute howToRunCompute=HowToRunCompute.REFLECT;
    public enum HowToRunKernel{REFLECT, BABYLON_INTERPRETER, BABYLON_CLASSFILE, LOWER_TO_SSA,LOWER_TO_SSA_AND_MAP_PTRS}
    HowToRunKernel howToRunKernel = HowToRunKernel.LOWER_TO_SSA_AND_MAP_PTRS;
    public DebugBackend(HowToRunCompute howToRunCompute, HowToRunKernel howToRunKernel){
        this.howToRunCompute = howToRunCompute;
        this.howToRunKernel = howToRunKernel;
    }

    @Override
    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        switch (howToRunCompute){
            case REFLECT: {
                try {
                    computeContext.computeCallGraph.entrypoint.method.invoke(null, args);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    throw new RuntimeException(e);
                }
                break;
            }
            case BABYLON_INTERPRETER:{
                if (computeContext.computeCallGraph.entrypoint.lowered == null) {
                    computeContext.computeCallGraph.entrypoint.lowered = computeContext.computeCallGraph.entrypoint.funcOpWrapper().lower();
                }
                Interpreter.invoke(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered.op(), args);
                break;
            }
            case BABYLON_CLASSFILE:{
                if (computeContext.computeCallGraph.entrypoint.lowered == null) {
                    computeContext.computeCallGraph.entrypoint.lowered = computeContext.computeCallGraph.entrypoint.funcOpWrapper().lower();
                }
                try {
                    if (computeContext.computeCallGraph.entrypoint.mh == null) {
                        computeContext.computeCallGraph.entrypoint.mh = BytecodeGenerator.generate(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered.op());
                    }
                    computeContext.computeCallGraph.entrypoint.mh.invokeWithArguments(args);
                } catch (Throwable e) {
                    computeContext.computeCallGraph.entrypoint.lowered.op().writeTo(System.out);
                    throw new RuntimeException(e);
                }
                break;
            }
        }
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {

        switch (howToRunKernel){
            case REFLECT: {
                KernelEntrypoint kernelEntrypoint = kernelCallGraph.entrypoint;
                for (ndRange.kid.x = 0; ndRange.kid.x < ndRange.kid.maxX; ndRange.kid.x++) {
                    try {
                        args[0] = ndRange.kid;
                        kernelEntrypoint.method.invoke(null, args);
                    } catch (IllegalAccessException e) {
                        throw new RuntimeException(e);
                    } catch (InvocationTargetException e) {
                        throw new RuntimeException(e);
                    }
                }
                break;
            }
            case BABYLON_INTERPRETER:{
                var lowered = kernelCallGraph.entrypoint.funcOpWrapper().lower();
                Interpreter.invoke(kernelCallGraph.computeContext.accelerator.lookup, lowered.op(), args);
                break;
            }
            case BABYLON_CLASSFILE:{
                var lowered = kernelCallGraph.entrypoint.funcOpWrapper().lower();
                var mh = BytecodeGenerator.generate(kernelCallGraph.computeContext.accelerator.lookup, lowered.op());
                try {
                    mh.invokeWithArguments(args);
                } catch (Throwable e) {
                    throw new RuntimeException(e);
                }
                break;
            }

            case LOWER_TO_SSA:{
                var highLevelForm = kernelCallGraph.entrypoint.method.getCodeModel().orElseThrow();


                System.out.println("Initial code model");
                System.out.println(highLevelForm.toText());
                System.out.println("------------------");

                CoreOp.FuncOp loweredForm = highLevelForm.transform(OpTransformer.LOWERING_TRANSFORMER);
                System.out.println("Lowered form which maintains original invokes and args");
                System.out.println(loweredForm.toText());
                System.out.println("-------------- ----");

                CoreOp.FuncOp ssaInvokeForm = SSA.transform(loweredForm);
                System.out.println("SSA form which maintains original invokes and args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

            }

            case LOWER_TO_SSA_AND_MAP_PTRS:{
                var highLevelForm = kernelCallGraph.entrypoint.method.getCodeModel().orElseThrow();
                System.out.println("Initial code model");
                System.out.println(highLevelForm.toText());
                System.out.println("------------------");
                CoreOp.FuncOp loweredForm = highLevelForm.transform(OpTransformer.LOWERING_TRANSFORMER);
                System.out.println("Lowered form which maintains original invokes and args");
                System.out.println(loweredForm.toText());
                System.out.println("-------------- ----");
                // highLevelForm.lower();
                CoreOp.FuncOp ssaInvokeForm = SSA.transform(loweredForm);
                System.out.println("SSA form which maintains original invokes and args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                FunctionType functionType = OpsAndTypes.transformTypes(MethodHandles.lookup(), ssaInvokeForm);
                System.out.println("SSA form with types transformed args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                CoreOp.FuncOp ssaPtrForm = OpsAndTypes.transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
                System.out.println("SSA form with invokes replaced by ptrs");
                System.out.println(ssaPtrForm.toText());
            }
        }
    }
}

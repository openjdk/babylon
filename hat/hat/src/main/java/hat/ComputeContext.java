package hat;

import hat.buffer.Buffer;
import hat.callgraph.ComputeCallGraph;
import hat.callgraph.KernelCallGraph;
import hat.optools.FuncOpWrapper;
import hat.optools.LambdaOpWrapper;
import hat.optools.OpWrapper;

import java.lang.reflect.Method;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.MethodRef;
import java.util.function.Consumer;

/**
 * A ComputeContext is created by an Accelerator to capture and control compute and kernel
 * callgraphs for the work to be performed by the backend.
 * <p/>
 * The Compute closure is created first, by walking the code model of the entrypoint, then transitively
 * visiting all conventional code reachable from this entrypoint.
 * <p/>
 * Generally all user defined methods reachable from the entrypoint (and the entrypoint intself) must be static methods of the same
 * enclosing classes.
 * <p/>
 * We do allow calls on the ComputeContext itself, and on the mapped interface buffers holding non uniform kernel data.
 * <p/>
 * Each request to dispatch a kernel discovered in the compute graph, results in a new Kernel call graph
 * being created with the dispatched kernel as it's entrypoint.
 * <p/>
 * When the ComputeContext is finalized, it is passed to the backend via <a href="Backend.computeClosureHandoff(ComputeContext)"></a>
 *
 * @author Gary Frost
 */
public class ComputeContext {
    public static final MethodRef M_CC_PRE_MUTATE = MethodRef.method(ComputeContext.class, "preMutate",
            void.class, Buffer.class);
    public static final MethodRef M_CC_POST_MUTATE = MethodRef.method(ComputeContext.class, "postMutate",
            void.class, Buffer.class);
    public static final MethodRef M_CC_PRE_ACCESS = MethodRef.method(ComputeContext.class, "preAccess",
            void.class, Buffer.class);
    public static final MethodRef M_CC_POST_ACCESS = MethodRef.method(ComputeContext.class, "postAccess",
            void.class, Buffer.class);
    public final Accelerator accelerator;


    public final ComputeCallGraph computeCallGraph;

    /**
     * Called by the Accelerator when the accelerator is passed a compute entrypoint.
     * <p>
     * So given a ComputeClass such as..
     * <pre>
     *  public class MyComputeClass {
     *    @ CodeReflection
     *    public static void addDeltaKernel(KernelContext kc, S32Array arrayOfInt, int delta) {
     *        arrayOfInt.array(kc.x, arrayOfInt.array(kc.x)+delta);
     *    }
     *
     *    @ CodeReflection
     *    static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
     *        cc.dispatchKernel(KernelContext kc -> addDeltaKernel(kc,arrayOfInt.length(), 5, arrayOfInt);
     *    }
     *  }
     *  </pre>
     *
     * @param accelerator
     * @param computeMethod
     */

    protected ComputeContext(Accelerator accelerator, Method computeMethod) {
        this.accelerator = accelerator;
        FuncOpWrapper funcOpWrapper = OpWrapper.wrap(computeMethod.getCodeModel().orElseThrow());
        this.computeCallGraph = new ComputeCallGraph(this, computeMethod, funcOpWrapper);

        this.computeCallGraph.close();
        this.accelerator.backend.computeContextHandoff(this);
    }

    /**
     * Called from within compute reachable code to dispatch a kernel.
     *
     * @param range
     * @param quotableKernelContextConsumer
     */

    public void dispatchKernel(int range, QuotableKernelContextConsumer quotableKernelContextConsumer) {
        Quoted quoted = quotableKernelContextConsumer.quoted();
        LambdaOpWrapper lambdaOpWrapper = OpWrapper.wrap((CoreOp.LambdaOp) quoted.op());
        MethodRef methodRef = lambdaOpWrapper.getQuotableTargetMethodRef();
        KernelCallGraph kernelCallGraph = computeCallGraph.kernelCallGraphMap.get(methodRef);
        try {
            Object[] args = lambdaOpWrapper.getQuotableCapturedValues(quoted, kernelCallGraph.entrypoint.method);
            NDRange ndRange = accelerator.range(range);
            args[0] = ndRange;
            accelerator.backend.dispatchKernel(kernelCallGraph, ndRange, args);
        } catch (Throwable t) {
            System.out.print("what?" + methodRef + " " + t);
            throw t;
        }
    }


    public void preMutate(Buffer b) {
        //System.out.println("preMutate " + b);
    }

    public void postMutate(Buffer b) {
        //System.out.println("postMutate " + b);
    }

    public void preAccess(Buffer b) {
        /*System.out.println("preAccess " + b);*/
    }

    public void postAccess(Buffer b) {
        /*System.out.println("postAccess " + b);*/
    }

    public interface QuotableKernelContextConsumer extends Quotable, Consumer<KernelContext> {


    }


}

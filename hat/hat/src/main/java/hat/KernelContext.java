package hat;

/**
 * Created by a dispatch call to a kernel from within a Compute method and 'conceptually' passed to a kernel.
 * <p>
 * In reality the backend decides how to pass the information contained in the KernelContext.
 *
 * <pre>
 *     @ CodeReflection
 *      static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
 *         cc.dispatchKernel(KernelContext kc -> addDeltaKernel(kc,arrayOfInt.length(), 5, arrayOfInt);
 *      }
 * </pre>
 *
 * @author Gary Frost
 */
public class KernelContext {
    public final NDRange ndRange;
    public int x;
    final public int maxX;

    public KernelContext(NDRange ndRange, int maxX, int x) {
        this.ndRange = ndRange;
        this.maxX = maxX;
        this.x = x;
    }

}

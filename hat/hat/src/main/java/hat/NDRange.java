package hat;

/**
 * Represents the range over a compute grid for a kernel to be applied.
 */
public class NDRange {
    public final Accelerator accelerator;

    public KernelContext kid;

    public NDRange(Accelerator accelerator) {
        this.accelerator = accelerator;
    }
}

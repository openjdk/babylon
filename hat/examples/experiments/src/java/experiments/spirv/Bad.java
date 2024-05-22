package experiments.spirv;

import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.util.function.Consumer;

public class Bad {
    public static class AcceleratorProxy {
        public interface QuotableComputeConsumer extends Quotable, Consumer<ComputeClosureProxy> {
        }

        public static class ComputeClosureProxy {
        }

        public void compute(AcceleratorProxy.QuotableComputeConsumer cqr) {
            Quoted quoted = cqr.quoted();
            Op op = quoted.op();
            System.out.println(op.toText());
        }

    }

    public static class MatrixMultiplyCompute {
        static void compute(AcceleratorProxy.ComputeClosureProxy computeContext, float[] a, float[] b, float[] c, int size) {}
    }

    //static final int size = 100; // works
    public static void main(String[] args) {
        AcceleratorProxy accelerator = new AcceleratorProxy();
        final int size = 100; // breaks!!!!
        //int size = 100;  // works
        var a = new float[]{};
        var b = new float[]{};
        var c = new float[]{};
        accelerator.compute(cc -> MatrixMultiplyCompute.compute(cc, a, b, c, size));
    }
}

package experiments;



import hat.Accelerator;
import hat.ComputeContext;
import hat.HatPtr;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.backend.BackendAdaptor;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import hat.ifacemapper.Schema;
import hat.backend.DebugBackend;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;
import java.lang.runtime.CodeReflection;

public class PointyHatArray {
    public interface PointArray extends Buffer {
        interface Point extends Struct {

            int x();

            void x(int x);

            int y();

            void y(int y);

            GroupLayout LAYOUT = MemoryLayout.structLayout(

                    ValueLayout.JAVA_INT.withName("x"),
                    ValueLayout.JAVA_INT.withName("y")
            );
        }

        int length();

        void length(int length);

        Point point(long idx);

        GroupLayout LAYOUT = MemoryLayout.structLayout(
                ValueLayout.JAVA_INT.withName("length"),
                MemoryLayout.sequenceLayout(100, Point.LAYOUT.withName(Point.class.getSimpleName())).withName("point")
        ).withName(PointArray.class.getSimpleName());


        Schema<PointArray> schema = Schema.of(PointArray.class, (pointArray)-> pointArray
                .arrayLen("length").array("point", (point)-> point

                                .field("x")
                                .field("y")
                )
        );

        static PointArray create(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator, int len) {
            System.out.println(LAYOUT);
            PointArray pointArray= schema.allocate(lookup,bufferAllocator,100);
            pointArray.length(100);
            return pointArray;
        }
    }

    public static class Compute {


        @CodeReflection
         public static void testMethodKernel(KernelContext kc, PointArray pointArray) {

            int len = pointArray.length();
            PointArray.Point point = pointArray.point(4);
            point.x(1);


        }

        @CodeReflection
        public static void compute(ComputeContext cc, PointArray pointArray) {
            cc.dispatchKernel(1, kc -> Compute.testMethodKernel(kc, pointArray));
        }

    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), new BackendAdaptor() {
            @Override
            public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
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

                FunctionType functionType = HatPtr.transformTypes(MethodHandles.lookup(), ssaInvokeForm);
                System.out.println("SSA form with types transformed args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                CoreOp.FuncOp ssaPtrForm = HatPtr.transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
                System.out.println("SSA form with invokes replaced by ptrs");
                System.out.println(ssaPtrForm.toText());
            }
        });
        var pointArray = PointArray.create(accelerator.lookup,accelerator,100);
        accelerator.compute(cc -> Compute.compute(cc, pointArray));
    }
}

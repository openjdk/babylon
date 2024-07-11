package experiments;



import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import hat.backend.DebugBackend;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
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

    static class Compute {


        @CodeReflection
        static void testMethodKernel(KernelContext kc, PointArray pointArray) {

            int len = pointArray.length();
            PointArray.Point point = pointArray.point(4);
            point.x(1);


        }

        @CodeReflection
        static void compute(ComputeContext cc, PointArray pointArray) {
            cc.dispatchKernel(1, kc -> Compute.testMethodKernel(kc, pointArray));
        }

    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), new DebugBackend(
                DebugBackend.HowToRunCompute.REFLECT,DebugBackend.HowToRunKernel.LOWER_TO_SSA_AND_MAP_PTRS));
        var pointArray = PointArray.create(accelerator.lookup,accelerator,100);
        accelerator.compute(cc -> Compute.compute(cc, pointArray));
    }
}

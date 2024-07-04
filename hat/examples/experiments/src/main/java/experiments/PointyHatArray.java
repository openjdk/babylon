package experiments;



import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.ifacemapper.Schema;
import hat.backend.DebugBackend;
import hat.buffer.BufferAllocator;
import hat.buffer.CompleteBuffer;
import hat.ifacemapper.HatData;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class PointyHatArray {
    public interface PointArray extends CompleteBuffer {
        interface Point extends StructChild {

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

        static PointArray create(BufferAllocator bufferAllocator, int len) {
            System.out.println(LAYOUT);
            System.out.println(schema.boundSchema(100).groupLayout);
            HatData hatData = new HatData() {
            };
            PointArray pointArray = bufferAllocator.allocate(SegmentMapper.of(MethodHandles.lookup(), PointArray.class, LAYOUT,hatData));
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
        var pointArray = PointArray.create(accelerator,100);
        accelerator.compute(cc -> Compute.compute(cc, pointArray));
    }
}

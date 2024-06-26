package experiments;



import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.Schema;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.buffer.CompleteBuffer;
import hat.ifacemapper.HatData;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class PointyHat {
    public interface ColoredWeightedPoint extends CompleteBuffer {

       public interface WeightedPoint extends Buffer.StructChild {
            int x();

            void x(int x);

            int y();

            void y(int y);

            float weight();

            void weight(float weight);

           GroupLayout LAYOUT = MemoryLayout.structLayout(
                    ValueLayout.JAVA_FLOAT.withName("weight"),
                    ValueLayout.JAVA_INT.withName("x"),
                    ValueLayout.JAVA_INT.withName("y")
            );
        }

        WeightedPoint weightedPoint();

        int color();

        void color(int v);

        GroupLayout LAYOUT = MemoryLayout.structLayout(
                WeightedPoint.LAYOUT.withName("weightedPoint"),
                ValueLayout.JAVA_INT.withName("color")
        ).withName(ColoredWeightedPoint.class.getSimpleName());



        Schema<ColoredWeightedPoint> schema = Schema.of(ColoredWeightedPoint.class, (cwp)-> cwp
                .field("weightedPoint", (wp)-> wp.fields("weight","x","y"))
                .field("color")
        );

        static ColoredWeightedPoint create(BufferAllocator bufferAllocator) {
            System.out.println(LAYOUT);
            System.out.println(schema.boundSchema().groupLayout);
            HatData hatData = new HatData() {
            };
            return bufferAllocator.allocate(SegmentMapper.of(MethodHandles.lookup(), ColoredWeightedPoint.class, LAYOUT,hatData));
        }
    }

    static class Compute {


        @CodeReflection
        static void testMethodKernel(KernelContext kc, ColoredWeightedPoint coloredWeightedPoint) {
            // StructOne* s1
            // s1 -> i
            int color = coloredWeightedPoint.color();
            // s1 -> *s2
            ColoredWeightedPoint.WeightedPoint weightedPoint = coloredWeightedPoint.weightedPoint();
            // s2 -> i
            color += weightedPoint.x();
            coloredWeightedPoint.color(color);
            // s2 -> f
            float weight = weightedPoint.weight();

        }

        @CodeReflection
        static void compute(ComputeContext cc, ColoredWeightedPoint coloredWeightedPoint) {
            cc.dispatchKernel(1, kc -> Compute.testMethodKernel(kc, coloredWeightedPoint));
        }

    }


    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), new PtrDebugBackend());
        var coloredWeightedPoint = ColoredWeightedPoint.create(accelerator);
        accelerator.compute(cc -> Compute.compute(cc, coloredWeightedPoint));
    }
}

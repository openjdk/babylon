package experiments;



import hat.Accelerator;
import hat.ComputeContext;
import hat.HatPtr;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.BackendAdaptor;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.ifacemapper.Schema;

import hat.buffer.BufferAllocator;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;
import java.lang.runtime.CodeReflection;

public class PointyHat {
    public interface ColoredWeightedPoint extends Buffer {
        interface WeightedPoint extends Struct {
            interface Point extends Struct {
                int x();

                void x(int x);

                int y();

                void y(int y);
            }

            float weight();

            void weight(float weight);

            Point point();
        }

        WeightedPoint weightedPoint();

        int color();

        void color(int v);

        Schema<ColoredWeightedPoint> schema = Schema.of(ColoredWeightedPoint.class, (colouredWeightedPoint)-> colouredWeightedPoint
                .field("weightedPoint", (weightedPoint)-> weightedPoint
                        .field("weight")
                        .field("point", point->point
                                .field("x")
                                .field("y"))
                )
                .field("color")
        );

        static ColoredWeightedPoint create(MethodHandles.Lookup lookup,BufferAllocator bufferAllocator) {
            return schema.allocate(lookup,bufferAllocator);
        }
    }

    public static class Compute {

        @CodeReflection
        public static void testMethodKernel(KernelContext kc, ColoredWeightedPoint coloredWeightedPoint) {
            // StructOne* s1
            // s1 -> i
            int color = coloredWeightedPoint.color();
            // s1 -> *s2
            ColoredWeightedPoint.WeightedPoint weightedPoint = coloredWeightedPoint.weightedPoint();
            // s2 -> i
            ColoredWeightedPoint.WeightedPoint.Point point = weightedPoint.point();
            color += point.x();
            coloredWeightedPoint.color(color);
            // s2 -> f
            float weight = weightedPoint.weight();

        }

        @CodeReflection
        public static void compute(ComputeContext cc, ColoredWeightedPoint coloredWeightedPoint) {
            cc.dispatchKernel(1, kc -> Compute.testMethodKernel(kc, coloredWeightedPoint));
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

                FunctionType functionType = HatPtr.transformTypes(MethodHandles.lookup(), ssaInvokeForm, args);
                System.out.println("SSA form with types transformed args");
                System.out.println(ssaInvokeForm.toText());
                System.out.println("------------------");

                CoreOp.FuncOp ssaPtrForm = HatPtr.transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
                System.out.println("SSA form with invokes replaced by ptrs");
                System.out.println(ssaPtrForm.toText());
            }
        });
        var coloredWeightedPoint = ColoredWeightedPoint.create(MethodHandles.lookup(),accelerator);

        int color = coloredWeightedPoint.color();
        // s1 -> *s2
        ColoredWeightedPoint.WeightedPoint weightedPoint = coloredWeightedPoint.weightedPoint();
        // s2 -> i
        ColoredWeightedPoint.WeightedPoint.Point point = weightedPoint.point();
        color += point.x();
        coloredWeightedPoint.color(color);
        // s2 -> f
        float weight = weightedPoint.weight();
        accelerator.compute(cc -> Compute.compute(cc, coloredWeightedPoint));
    }
}

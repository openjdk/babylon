package experiments;


import hat.HatPtr;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FunctionType;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

import  hat.HatPtr;

public class InvokeToPtr {

    @CodeReflection
    static float testMethod(PointyHat.ColoredWeightedPoint coloredWeightedPoint) {
        // StructOne* s1
        // s1 -> i
        int color = coloredWeightedPoint.color();
        // s1 -> *s2
        PointyHat.ColoredWeightedPoint.WeightedPoint weightedPoint = coloredWeightedPoint.weightedPoint();
        // s2 -> i
        PointyHat.ColoredWeightedPoint.WeightedPoint.Point point = weightedPoint.point();
        color += point.x();
        coloredWeightedPoint.color(color);
        // s2 -> f
        float weight = weightedPoint.weight();
        return color + weight;
    }


    public static void main(String[] args) {
        System.out.println(PointyHat.ColoredWeightedPoint.LAYOUT);
        System.out.println(PointyHat.ColoredWeightedPoint.schema.boundSchema().groupLayout);
        Optional<Method> om = Stream.of(InvokeToPtr.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("testMethod"))
                .findFirst();

        Method m = om.orElseThrow();
        CoreOp.FuncOp highLevelForm = m.getCodeModel().orElseThrow();

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

        FunctionType functionType = HatPtr.transformTypes(MethodHandles.lookup(), ssaInvokeForm);
        System.out.println("SSA form with types transformed args");
        System.out.println(ssaInvokeForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp ssaPtrForm = HatPtr.transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
        System.out.println("SSA form with invokes replaced by ptrs");
        System.out.println(ssaPtrForm.toText());
    }
}

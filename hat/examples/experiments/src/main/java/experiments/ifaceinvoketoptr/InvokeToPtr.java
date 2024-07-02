package experiments.ifaceinvoketoptr;


import hat.Schema;
import hat.buffer.Buffer;
import hat.buffer.CompleteBuffer;
import hat.buffer.MappableIface;
import hat.optools.InvokeOpWrapper;
import hat.optools.OpWrapper;

import java.lang.constant.ClassDesc;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

public class InvokeToPtr {
    public interface ColoredWeightedPoint extends CompleteBuffer {

        interface WeightedPoint extends Buffer.StructChild {
            int x();

            void x(int x);

            int y();

            void y(int y);

            float weight();

            void weight(float weight);

            MemoryLayout LAYOUT = MemoryLayout.structLayout(
                    ValueLayout.JAVA_FLOAT.withName("weight"),
                    ValueLayout.JAVA_INT.withName("x"),
                    ValueLayout.JAVA_INT.withName("y")
            );
        }

        WeightedPoint weightedPoint();

        int color();

        void color(int color);

        MemoryLayout LAYOUT = MemoryLayout.structLayout(
                WeightedPoint.LAYOUT.withName("weightedPoint"),
                ValueLayout.JAVA_INT.withName("color")
        ).withName(ColoredWeightedPoint.class.getSimpleName());

        Schema<ColoredWeightedPoint> schema = Schema.of(ColoredWeightedPoint.class, (cwp) -> cwp
                .field("weightedPoint", (wp) -> wp.fields("weight", "x", "y"))
                .field("color")
        );
    }

    @CodeReflection
    static float testMethod(ColoredWeightedPoint coloredWeightedPoint) {
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
        return color + weight;
    }


    public static void main(String[] args) {
        System.out.println(ColoredWeightedPoint.LAYOUT);
        System.out.println(ColoredWeightedPoint.schema.boundSchema().groupLayout);
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

        FunctionType functionType = transformTypes(MethodHandles.lookup(), ssaInvokeForm);
        System.out.println("SSA form with types transformed args");
        System.out.println(ssaInvokeForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp ssaPtrForm = transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm, functionType);
        System.out.println("SSA form with invokes replaced by ptrs");
        System.out.println(ssaPtrForm.toText());
    }

    static <T extends MappableIface> Class<T> getMappableClassOrNull(MethodHandles.Lookup lookup, TypeElement typeElement) {
        try {
            return (typeElement instanceof JavaType jt
                    && jt.resolve(lookup) instanceof Class<?> possiblyMappableIface
                    && MappableIface.class.isAssignableFrom(possiblyMappableIface)) ? (Class<T>) possiblyMappableIface : null;
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    static <T extends MappableIface> TypeElement convertToPtrTypeIfPossible(MethodHandles.Lookup lookup, TypeElement typeElement) {
        return getMappableClassOrNull(lookup, typeElement) instanceof Class<?> clazz
                ? new PtrType<>((Class<T>) clazz, getLayout((Class<T>) clazz))
                : typeElement;
    }


    static FunctionType transformTypes(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        List<TypeElement> transformedTypeElements = new ArrayList<>();
        for (Block.Parameter parameter : funcOp.parameters()) {
            transformedTypeElements.add(convertToPtrTypeIfPossible(lookup, parameter.type()));
        }
        return FunctionType.functionType(convertToPtrTypeIfPossible(lookup, funcOp.invokableType().returnType()), transformedTypeElements);
    }

    static <T extends MappableIface> CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup lookup,
                                                                          CoreOp.FuncOp ssaForm, FunctionType functionType) {
        return CoreOp.func(ssaForm.funcName(), functionType).body(funcBlock -> {
            funcBlock.transformBody(ssaForm.body(), funcBlock.parameters(), (builder, op) -> {
                /*
                   We are looking for
                      interface Iface extends Buffer // or Buffer.StructChild
                         T foo();
                         void foo(T foo);
                      }
                   Were T is either a primitive or a nested iface mapping and foo matches the field name
                 */

                if (op instanceof CoreOp.InvokeOp invokeOp
                        && OpWrapper.wrap(invokeOp) instanceof InvokeOpWrapper invokeOpWrapper
                        && invokeOpWrapper.hasOperands()
                        && invokeOpWrapper.isIfaceBufferMethod()
                        && invokeOpWrapper.getReceiver() instanceof Value iface // Is there a containing iface type Iface
                        && getMappableClassOrNull(lookup, iface.type()) != null
                ) {
                    Value ifaceValue = builder.context().getValue(iface);     // ? Ensure we have an output value for the iface
                    PtrOp<T> ptrOp = new PtrOp<>(ifaceValue, invokeOpWrapper.name());         // Create ptrOp to replace invokeOp
                    Op.Result ptrResult = builder.op(ptrOp);// replace and capture the result of the invoke
                    if (invokeOpWrapper.operandCount() == 1) {                  // No args (operand(0)==containing iface))
                        /*
                          this turns into a load
                          interface Iface extends Buffer // or Buffer.StructChild
                              T foo();
                          }
                         */
                        if (ptrOp.resultType().layout() instanceof ValueLayout) { // are we pointing to a primitive
                            PtrLoadValue primitiveLoad = new PtrLoadValue(iface.type(), ptrResult);
                            Op.Result replacedReturnValue = builder.op(primitiveLoad);
                            builder.context().mapValue(invokeOp.result(), replacedReturnValue);
                        } else {                                                 // pointing to another iface mappable
                            builder.context().mapValue(invokeOp.result(), ptrResult);
                        }
                    } else if (invokeOpWrapper.operandCount() == 2) {
                         /*
                          This turns into a store
                          interface Iface extends Buffer // or Buffer.StructChild
                              void foo(T);
                          }
                         */
                        if (ptrOp.resultType().layout() instanceof ValueLayout) { // are we pointing to a primitive
                            Value valueToStore = builder.context().getValue(invokeOpWrapper.operandNAsValue(1));
                            PtrStoreValue primitiveStore = new PtrStoreValue(iface.type(), ptrResult, valueToStore);
                            Op.Result replacedReturnValue = builder.op(primitiveStore);
                            builder.context().mapValue(invokeOp.result(), replacedReturnValue);
                        } else {                                                 // pointing to another iface mappable
                            builder.context().mapValue(invokeOp.result(), ptrResult);
                        }
                    } else {
                        builder.op(op);
                    }
                } else {
                    builder.op(op);
                }
                return builder; // why? oh why?
            });
        });
    }

    static <T extends MappableIface> MemoryLayout getLayout(Class<T> mappableIface) {
        try {
            return (MemoryLayout) mappableIface.getDeclaredField("LAYOUT").get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static final class PtrType<T extends MappableIface> implements TypeElement {
        static final String NAME = "ptrType";
        final Class<T> mappableIface;
        final MemoryLayout layout;
        final JavaType referringType;

        public PtrType(Class<T> mappableIface, MemoryLayout layout) {
            this.mappableIface = mappableIface;
            this.layout = layout;
            this.referringType = switch (layout) {
                case StructLayout _ -> JavaType.type(ClassDesc.of(layout.name().orElseThrow()));
                case AddressLayout _ -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
                case ValueLayout valueLayout -> JavaType.type(valueLayout.carrier());
                default -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            };
        }

        public JavaType referringType() {
            return referringType;
        }

        public MemoryLayout layout() {
            return layout;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            PtrType<T> ptrType = (PtrType<T>) o;
            return Objects.equals(layout, ptrType.layout);
        }

        @Override
        public int hashCode() {
            return Objects.hash(layout);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(referringType.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    }

    @OpFactory.OpDeclaration(PtrOp.NAME)
    public static final class PtrOp<T extends MappableIface> extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        final PtrType<T> ptrType;
        final PtrType<T> resultType;
        final String simpleMemberName;
        final long memberOffset;


        PtrOp(PtrOp<T> that, CopyContext cc) {
            super(that, cc);
            this.ptrType = that.ptrType;
            this.resultType = that.resultType;
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;

        }

        @Override
        public PtrOp<T> transform(CopyContext cc, OpTransformer ot) {
            return new PtrOp<T>(this, cc);
        }

        public PtrOp(Value ptr, String simpleMemberName) {
            super(NAME, List.of(ptr));

            this.simpleMemberName = simpleMemberName;
            if (ptr.type() instanceof PtrType<?> ptrType) {

                if (ptrType.layout() instanceof StructLayout structLayout) {
                    this.ptrType = (PtrType<T>) ptrType;
                    MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(simpleMemberName);
                    this.memberOffset = structLayout.byteOffset(memberPathElement);
                    MemoryLayout memberLayout = structLayout.select(memberPathElement);
                    // So we need a type for simpleMemberName ?


                    //   Arrays.stream(ptrType.mappableIface.getDeclaredMethods()).forEach(m->{

                    //    System.out.println(simpleMemberName+" "+memberLayout.name() + " "+m.getName()+" "+m.getReturnType());
                    //  });
                    this.resultType = new PtrType<>((Class<T>) ptrType.mappableIface, memberLayout);
                } else {
                    throw new IllegalArgumentException("Pointer type layout is not a struct layout: " + ptrType.layout());
                }
            } else {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
        }


        @Override
        public PtrType<T> resultType() {
            return resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", simpleMemberName);
            attrs.put(ATTRIBUTE_OFFSET, memberOffset);
            return attrs;
        }
    }


    public static abstract class PtrAccessValue extends Op {
        final String name;
        final JavaType resultType;
        final TypeElement typeElement;

        PtrAccessValue(String name, TypeElement typeElement, PtrAccessValue that, CopyContext cc) {
            super(that, cc);
            this.name = name;
            this.typeElement = typeElement;
            this.resultType = that.resultType;
        }

        public PtrAccessValue(String name, TypeElement typeElement, JavaType resultType, List<Value> values) {
            super(name, values);
            this.name = name;
            this.typeElement = typeElement;
            this.resultType = resultType;
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }
    }

    @OpFactory.OpDeclaration(PtrLoadValue.NAME)
    public static final class PtrLoadValue extends PtrAccessValue {
        public static final String NAME = "ptr.load.value";

        PtrLoadValue(TypeElement typeElement, PtrLoadValue that, CopyContext cc) {
            super(NAME, typeElement, that, cc);
        }

        @Override
        public PtrLoadValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrLoadValue(typeElement, this, cc);
        }

        public PtrLoadValue(TypeElement typeElement, Value ptr) {
            super(NAME, typeElement, ((PtrType<?>) ptr.type()).referringType(), List.of(ptr));
        }

    }

    @OpFactory.OpDeclaration(PtrStoreValue.NAME)
    public static final class PtrStoreValue extends PtrAccessValue {
        public static final String NAME = "ptr.store.value";

        PtrStoreValue(TypeElement typeElement, PtrStoreValue that, CopyContext cc) {
            super(NAME, typeElement, that, cc);
        }

        @Override
        public PtrStoreValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrStoreValue(typeElement, this, cc);
        }

        public PtrStoreValue(TypeElement typeElement, Value ptr, Value arg1) {
            super(NAME, typeElement, JavaType.VOID, List.of(ptr, arg1));
        }
    }

}

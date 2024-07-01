package experiments.ifaceinvoketoptr;


import hat.Schema;
import hat.buffer.Buffer;
import hat.buffer.MappableIface;

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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class InvokeToPtr {
    public interface ColoredWeightedPoint extends Buffer {

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

        void color(int v);

        MemoryLayout LAYOUT = MemoryLayout.structLayout(
                WeightedPoint.LAYOUT.withName(WeightedPoint.class.getName() + "::weightedPoint"),
                ValueLayout.JAVA_INT.withName("color")
        ).withName(ColoredWeightedPoint.class.getName());

        Schema<ColoredWeightedPoint> schema = Schema.of(ColoredWeightedPoint.class, (cwp)-> cwp
                .field("weightedPoint", (wp)-> wp.fields("weight","x","y"))
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

        CoreOp.FuncOp ssaInvokeForm = SSA.transform(highLevelForm);
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

    static Class<?> getMappableClassOrNull(MethodHandles.Lookup lookup, TypeElement typeElement) {
        try {
            return (typeElement instanceof JavaType jt
                    && jt.resolve(lookup) instanceof Class<?> c
                    &&  MappableIface.class.isAssignableFrom(c))?c:null;
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

   static TypeElement convertToPtrTypeIfPossible(MethodHandles.Lookup lookup, TypeElement typeElement){
        return getMappableClassOrNull(lookup, typeElement) instanceof Class<?> clazz
                ?new PtrType(getLayout(clazz))
                :typeElement;
    }


    static FunctionType transformTypes(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        List<TypeElement> transformedTypeElements = new ArrayList<>();
        for (Block.Parameter parameter : funcOp.parameters()) {
            transformedTypeElements.add(convertToPtrTypeIfPossible(lookup,parameter.type()));
        }
        return FunctionType.functionType(convertToPtrTypeIfPossible(lookup,funcOp.invokableType().returnType()), transformedTypeElements);
    }

    static CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup lookup,
                                                CoreOp.FuncOp ssaForm, FunctionType functionType) {
        return CoreOp.func(ssaForm.funcName(), functionType).body(funcBlock -> {
            funcBlock.transformBody(ssaForm.body(), funcBlock.parameters(), (builder, op) -> {
                /*
                   We are looking for
                      interface Iface extends Buffer // or Buffer.StructChild
                         T foo();
                      }
                   Were T is either a primitive or a nested iface mapping and foo matches the field name
                 */
                if (       op instanceof CoreOp.InvokeOp invokeOp
                        && invokeOp.hasReceiver()                                 // Is there a containing iface type Iface
                        && invokeOp.operands().size() == 1                        // No args (operand(0)==containing iface)
                        && invokeOp.operands().getFirst() instanceof Value iface  // Get the containing iface
                        && getMappableClassOrNull(lookup, iface.type()) != null        // check it is indeed mappable
                ) {
                        Value ifaceValue = builder.context().getValue(iface);     // ? Ensure we have an output value for the iface

                        String methodName =  invokeOp.invokeDescriptor().name();  // foo in our case
                        PtrOp ptrOp = new PtrOp( ifaceValue, methodName);         // Create ptrOp to replace invokeOp
                        Op.Result ptrResult = builder.op(ptrOp);                  // replace and capture the result of the invoke

                        if (ptrOp.resultType().layout() instanceof ValueLayout) { // are we pointing to a primitive
                            PtrLoadValue primitiveLoad = new PtrLoadValue(ptrResult);
                            Op.Result replacedReturnValue = builder.op(primitiveLoad);
                            builder.context().mapValue(invokeOp.result(),replacedReturnValue);
                        } else {                                                 // pointing to another iface mappable
                            builder.context().mapValue(invokeOp.result(), ptrResult);
                        }
                } else {
                    builder.op(op);
                }
                return builder; // why? oh why?
            });
        });
    }

    static MemoryLayout getLayout(Class<?> clazz){
        try {
            return (MemoryLayout) clazz.getDeclaredField("LAYOUT").get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    public static final class PtrType implements TypeElement {
        static final String NAME = "ptr";
        final MemoryLayout layout;
        final JavaType referringType;

        public PtrType(MemoryLayout layout) {
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
            PtrType ptrType = (PtrType) o;
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
    public static final class PtrOp extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        public static final String ATTRIBUTE_CONTAINED_BY = "c";
        final String simpleMemberName;
        final long memberOffset;
        final PtrType resultType;

        PtrOp(PtrOp that, CopyContext cc) {
            super(that, cc);
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;
            this.resultType = that.resultType;
        }

        @Override
        public PtrOp transform(CopyContext cc, OpTransformer ot) {
            return new PtrOp(this, cc);
        }

        public PtrOp(Value ptr, String simpleMemberName) {
            super(NAME, List.of(ptr));
            this.simpleMemberName = simpleMemberName;

            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            // @@@ Support group layout
            if (!(ptrType.layout() instanceof StructLayout structLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a struct layout: " + ptrType.layout());
            }

            // Find the actual member name from the simple member name
            String memberName = structLayout.memberLayouts().stream()
                            .map(layout -> layout.name().orElseThrow())
                            .filter(name -> (regex.matcher(name) instanceof Matcher matcher && matcher.matches() ? matcher.group(2) : name)
                                    .equals(simpleMemberName)) // foo::bar -> bar
                            .findFirst().orElseThrow();

            MemoryLayout.PathElement memberPathElement = MemoryLayout.PathElement.groupElement(memberName);
            this.memberOffset = structLayout.byteOffset(memberPathElement);
            MemoryLayout memberLayout = structLayout.select(memberPathElement);
            // Remove any simple member name from the layout

            MemoryLayout ptrLayout = memberLayout instanceof StructLayout
                    ? memberLayout.withName(regex.matcher(memberName) instanceof Matcher matcher && matcher.matches()
                            ? matcher.group(1) : null) // foo::bar -> foo
                    : memberLayout.withoutName();
            this.resultType = new PtrType(ptrLayout);
        }


        static Pattern regex = Pattern.compile("(.*)::(.*)");

        @Override
        public PtrType resultType() {
            return resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", simpleMemberName);
            attrs.put(ATTRIBUTE_OFFSET, memberOffset);
            attrs.put(ATTRIBUTE_CONTAINED_BY, "grf");
            return attrs;
        }
    }


    @OpFactory.OpDeclaration(PtrOp.NAME)
    public static final class PtrLoadValue extends Op {
        public static final String NAME = "ptr.load.value";
        final JavaType resultType;

        PtrLoadValue(PtrLoadValue that, CopyContext cc) {
            super(that, cc);
            this.resultType = that.resultType;
        }

        @Override
        public PtrLoadValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrLoadValue(this, cc);
        }

        public PtrLoadValue(Value ptr) {
            super(NAME, List.of(ptr));
            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(ptrType.layout() instanceof ValueLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a value layout: " + ptrType.layout());
            }
            this.resultType = ptrType.referringType();
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }

    }

    @OpFactory.OpDeclaration(PtrOp.NAME)
    public static final class PtrStoreValue extends Op {
        public static final String NAME = "ptr.store.value";

        PtrStoreValue(PtrStoreValue that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PtrStoreValue transform(CopyContext cc, OpTransformer ot) {
            return new PtrStoreValue(this, cc);
        }

        public PtrStoreValue(Value ptr, Value v) {
            super(NAME, List.of(ptr));

            if (!(ptr.type() instanceof PtrType ptrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(ptrType.layout() instanceof ValueLayout)) {
                throw new IllegalArgumentException("Pointer type layout is not a value layout: " + ptrType.layout());
            }
            if (!(ptrType.referringType().equals(v.type()))) {
                throw new IllegalArgumentException("Pointer reference type is not same as value to store type: "
                        + ptrType.referringType() + " " + v.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }
    }

}

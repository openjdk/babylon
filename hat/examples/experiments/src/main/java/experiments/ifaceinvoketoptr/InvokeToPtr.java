package experiments.ifaceinvoketoptr;


import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.constant.ClassDesc;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExternalizableOp;
import java.lang.reflect.code.op.OpFactory;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.PrimitiveType;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

public class InvokeToPtr {

    /*
    struct ColoredWeightedPoint{
       struct WeightedPoint{
         int x;
         int y;
         float weight;
       }
       struct WeightedPoint weightedPoint;
       int color;
     }
     */
    @Struct
    public interface ColoredWeightedPoint {

        @Struct
        public interface WeightedPoint {
            int x();
            void x(int x);
            int y();
            void y(int y);
            float weight();
            void weight(float weight);
            static MemoryLayout layout() {
                return LAYOUT;
            }
            MemoryLayout LAYOUT = MemoryLayout.structLayout(
                    ValueLayout.JAVA_FLOAT.withName("weight"),
                            ValueLayout.JAVA_INT.withName("x"),
                      ValueLayout.JAVA_INT.withName("y"))
                    .withName(WeightedPoint.class.getName());
        }

        WeightedPoint weightedPoint();
        int color();
        void color(int v);

        static MemoryLayout layout() {
            return LAYOUT;
        }
        MemoryLayout LAYOUT = MemoryLayout.structLayout(
                        WeightedPoint.layout().withName(WeightedPoint.class.getName() + "::weightedPoint"),
                        ValueLayout.JAVA_INT.withName("color"))
                .withName(ColoredWeightedPoint.class.getName());
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
        // s2 -> f
        float weight = weightedPoint.weight();
        return color + weight;
    }


    public static void main(String[] args) {
        CoreOp.FuncOp highLevelForm = getFuncOp("testMethod");

        System.out.println("Initial code model");
        System.out.println(highLevelForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp ssaInvokeForm = SSA.transform(highLevelForm);
        System.out.println("SSA form which maintains original invokes and args");
        System.out.println(ssaInvokeForm.toText());
        System.out.println("------------------");

        CoreOp.FuncOp ssaPtrForm = transformInvokesToPtrs(MethodHandles.lookup(), ssaInvokeForm);
        System.out.println("SSA form with invokes replaced by ptrs");
        System.out.println(ssaPtrForm.toText());
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(InvokeToPtr.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

    //

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.TYPE)
    public @interface Struct {
    }

    static CoreOp.FuncOp transformInvokesToPtrs(MethodHandles.Lookup l,
                                                CoreOp.FuncOp f) {
        List<TypeElement> pTypes = new ArrayList<>();
        for (Block.Parameter p : f.parameters()) {
            pTypes.add(transformStructClassToPtr(l, p.type()));
        }
        FunctionType functionType = FunctionType.functionType(
                transformStructClassToPtr(l, f.invokableType().returnType()),
                pTypes);
        return CoreOp.func(f.funcName(), functionType).body(funcBlock -> {
            funcBlock.transformBody(f.body(), funcBlock.parameters(), (b, op) -> {
                if (op instanceof CoreOp.InvokeOp iop && iop.hasReceiver()) {
                    Value receiver = iop.operands().getFirst();
                    if (structClass(l, receiver.type()) instanceof Class<?> _) {
                        Value ptr = b.context().getValue(receiver);
                        PtrToMember ptrToMemberOp = new PtrToMember(ptr, iop.invokeDescriptor().name());
                        Op.Result memberPtr = b.op(ptrToMemberOp);

                        if (iop.operands().size() == 1) {
                            // Pointer access and (possibly) value load
                            if (ptrToMemberOp.resultType().layout() instanceof ValueLayout) {
                                Op.Result v = b.op(new PtrLoadValue(memberPtr));
                                b.context().mapValue(iop.result(), v);
                            } else {
                                b.context().mapValue(iop.result(), memberPtr);
                            }
                        } else {
                            // @@@
                            // Value store
                            throw new UnsupportedOperationException();
                        }
                    } else {
                        b.op(op);
                    }
                } else {
                    b.op(op);
                }
                return b;
            });
        });
    };


    static TypeElement transformStructClassToPtr(MethodHandles.Lookup l, TypeElement type) {
        if (structClass(l, type) instanceof Class<?> sc) {
            return new PtrType(structClassLayout(l, sc));
        } else {
            return type;
        }
    }

    static MemoryLayout structClassLayout(MethodHandles.Lookup l,
                                          Class<?> c) {
        if (!c.isAnnotationPresent(Struct.class)) {
            throw new IllegalArgumentException();
        }

        Method layoutMethod;
        try {
            layoutMethod = c.getMethod("layout");
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
        MethodHandle layoutHandle;
        try {
            layoutHandle = l.unreflect(layoutMethod);
        } catch (IllegalAccessException e) {
            throw new RuntimeException(e);
        }
        try {
            return (MemoryLayout) layoutHandle.invoke();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    static Class<?> structClass(MethodHandles.Lookup l, TypeElement t) {
        try {
            return _structClass(l, t);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }
    static Class<?> _structClass(MethodHandles.Lookup l, TypeElement t) throws ReflectiveOperationException {
        if (!(t instanceof JavaType jt) || !(jt.resolve(l) instanceof Class<?> c)) {
            return null;
        }

        return c.isInterface() && c.isAnnotationPresent(Struct.class) ? c : null;
    }


    public static final class PtrType implements TypeElement {
        static final String NAME = "ptr";
        final MemoryLayout layout;
        final JavaType rType;

        public PtrType(MemoryLayout layout) {
            this.layout = layout;
            this.rType = switch (layout) {
                case StructLayout _ -> JavaType.type(ClassDesc.of(layout.name().orElseThrow()));
                case AddressLayout _ -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
                case ValueLayout valueLayout -> JavaType.type(valueLayout.carrier());
                default -> throw new UnsupportedOperationException("Unsupported member layout: " + layout);
            };
        }

        public JavaType rType() {
            return rType;
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
            return new ExternalizedTypeElement(NAME, List.of(rType.externalize()));
        }

        @Override
        public String toString() {
            return externalize().toString();
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrToMember extends ExternalizableOp {
        public static final String NAME = "ptr.to.member";
        public static final String ATTRIBUTE_OFFSET = "offset";
        public static final String ATTRIBUTE_NAME = "name";

        final String simpleMemberName;
        final long memberOffset;
        final PtrType resultType;

        PtrToMember(PtrToMember that, CopyContext cc) {
            super(that, cc);
            this.simpleMemberName = that.simpleMemberName;
            this.memberOffset = that.memberOffset;
            this.resultType = that.resultType;
        }

        @Override
        public PtrToMember transform(CopyContext cc, OpTransformer ot) {
            return new PtrToMember(this, cc);
        }

        public PtrToMember(Value ptr, String simpleMemberName) {
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
            String memberName = findMemberName(structLayout, simpleMemberName);
            MemoryLayout.PathElement p = MemoryLayout.PathElement.groupElement(memberName);
            this.memberOffset = structLayout.byteOffset(p);
            MemoryLayout memberLayout = structLayout.select(p);
            // Remove any simple member name from the layout
            MemoryLayout ptrLayout = memberLayout instanceof StructLayout
                    ? memberLayout.withName(className(memberName))
                    : memberLayout.withoutName();
            this.resultType = new PtrType(ptrLayout);
        }

        // @@@ Change to return member index
        static String findMemberName(StructLayout sl, String simpleMemberName) {
            for (MemoryLayout layout : sl.memberLayouts()) {
                String memberName = layout.name().orElseThrow();
                if (simpleMemberName(memberName).equals(simpleMemberName)) {
                    return memberName;
                }
            }
            throw new NoSuchElementException("No member found: " + simpleMemberName + " " + sl);
        }

        static String simpleMemberName(String memberName) {
            int i = memberName.indexOf("::");
            return i != -1
                    ? memberName.substring(i + 2)
                    : memberName;
        }

        static String className(String memberName) {
            int i = memberName.indexOf("::");
            return i != -1
                    ? memberName.substring(0, i)
                    : null;
        }

        @Override
        public PtrType resultType() {
            return resultType;
        }

        @Override
        public Map<String, Object> attributes() {
            HashMap<String, Object> attrs = new HashMap<>(super.attributes());
            attrs.put("", simpleMemberName);
            attrs.put(ATTRIBUTE_OFFSET, memberOffset);
            return attrs;
        }

        public String simpleMemberName() {
            return simpleMemberName;
        }

        public long memberOffset() {
            return memberOffset;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }


    @OpFactory.OpDeclaration(PtrToMember.NAME)
    public static final class PtrAddOffset extends Op {
        public static final String NAME = "ptr.add.offset";

        PtrAddOffset(PtrAddOffset that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public PtrAddOffset transform(CopyContext cc, OpTransformer ot) {
            return new PtrAddOffset(this, cc);
        }

        public PtrAddOffset(Value ptr, Value offset) {
            super(NAME, List.of(ptr, offset));

            if (!(ptr.type() instanceof PtrType)) {
                throw new IllegalArgumentException("Pointer value is not of pointer type: " + ptr.type());
            }
            if (!(offset.type() instanceof PrimitiveType pt && pt.equals(JavaType.LONG))) {
                throw new IllegalArgumentException("Offset value is not of primitve long type: " + offset.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return ptrValue().type();
        }

        public Value ptrValue() {
            return operands().get(0);
        }

        public Value offsetValue() {
            return operands().get(1);
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
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
            this.resultType = ptrType.rType();
        }

        @Override
        public TypeElement resultType() {
            return resultType;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }

    @OpFactory.OpDeclaration(PtrToMember.NAME)
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
            if (!(ptrType.rType().equals(v.type()))) {
                throw new IllegalArgumentException("Pointer reference type is not same as value to store type: "
                        + ptrType.rType() + " " + v.type());
            }
        }

        @Override
        public TypeElement resultType() {
            return JavaType.VOID;
        }

        public Value ptrValue() {
            return operands().get(0);
        }
    }
}

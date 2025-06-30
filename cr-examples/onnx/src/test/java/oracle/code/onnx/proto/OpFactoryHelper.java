package oracle.code.onnx.proto;

import jdk.incubator.code.Op;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public final class OpFactoryHelper {

    /**
     * A class value for lazily computing an operation factory for {@link Op operation} classes
     * annotated with {@link OpFactory.OpDeclaration} and enclosed within a given class to compute over.
     * <p>
     * Each enclosed class annotated with {@code OpDeclaration} must declare a public static method named {@code create}
     * with one parameter type of {@link ExternalizedOp} and return type that is the concrete class type.
     * Alternatively, the concrete class must declare public constructor with one parameter type of
     * {@link ExternalizedOp}.
     */
    static final ClassValue<OpFactory> OP_FACTORY = new ClassValue<>() {
        @Override
        protected OpFactory computeValue(Class<?> c) {
            // @@@ See https://bugs.openjdk.org/browse/JDK-8321207
            final Map<String, Class<? extends Op>> opMapping = createOpMapping(c);

            return def -> {
                var opClass = opMapping.get(def.name());
                if (opClass == null) {
                    return null;
                }

                Op op = constructOp(opClass, def);
                // Set location if available
                if (op != null && def.location() != null) {
                    op.setLocation(def.location());
                }
                return op;
            };
        }
    };

    private static Map<String, Class<? extends Op>> createOpMapping(Class<?> opClasses) {
        Map<String, Class<? extends Op>> mapping = new HashMap<>();
        for (Class<?> opClass : opClasses.getNestMembers()) {
            if (opClass.isAnnotationPresent(OpFactory.OpDeclaration.class)) {
                if (!Modifier.isPublic(opClass.getModifiers())) {
                    throw new InternalError("Operation class not public: " + opClass.getName());
                }

                if (!Op.class.isAssignableFrom(opClass)) {
                    throw new InternalError("Operation class is not assignable to Op: " + opClass);
                }

                MethodHandle handle = getOpConstructorMethodHandle(opClass);
                if (handle == null) {
                    throw new InternalError("Operation constructor for operation class not found: " + opClass.getName());
                }

                if (!Op.class.isAssignableFrom(handle.type().returnType())) {
                    throw new InternalError("Operation constructor does not return an Op: " + handle);
                }

                String opName = opClass.getAnnotation(OpFactory.OpDeclaration.class).value();
                @SuppressWarnings("unchecked")
                var opClassCast = (Class<Op>) opClass;
                mapping.put(opName, opClassCast);
            }
        }
        return mapping;
    }

    private static MethodHandle getOpConstructorMethodHandle(Class<?> opClass) {
        Method method = null;
        try {
            method = opClass.getMethod("create", ExternalizedOp.class);
        } catch (NoSuchMethodException e) {
        }

        if (method != null) {
            if (!Modifier.isStatic(method.getModifiers())) {
                throw new InternalError("Operation constructor is not a static method: " + method);
            }

            try {
                return MethodHandles.publicLookup().unreflect(method);
            } catch (IllegalAccessException e) {
                throw new InternalError("Inaccessible operation constructor for operation: " +
                        method);
            }
        }

        Constructor<?> constructor;
        try {
            constructor = opClass.getConstructor(ExternalizedOp.class);
        } catch (NoSuchMethodException e) {
            return null;
        }

        try {
            return MethodHandles.publicLookup().unreflectConstructor(constructor);
        } catch (IllegalAccessException e) {
            throw new InternalError("Inaccessible operation constructor for operation: " +
                    constructor);
        }
    }

    private static Op constructOp(Class<? extends Op> opClass, ExternalizedOp opDef) {
        class Enclosed {
            private static final ClassValue<Function<ExternalizedOp, Op>> OP_CONSTRUCTOR = new ClassValue<>() {
                @Override
                protected Function<ExternalizedOp, Op> computeValue(Class<?> opClass) {
                    final MethodHandle opConstructorMH = getOpConstructorMethodHandle(opClass);
                    assert opConstructorMH != null;

                    return operationDefinition -> {
                        try {
                            return (Op) opConstructorMH.invoke(operationDefinition);
                        } catch (RuntimeException | Error e) {
                            throw e;
                        } catch (Throwable t) {
                            throw new RuntimeException(t);
                        }
                    };
                }
            };
        }
        return Enclosed.OP_CONSTRUCTOR.get(opClass).apply(opDef);
    }
}

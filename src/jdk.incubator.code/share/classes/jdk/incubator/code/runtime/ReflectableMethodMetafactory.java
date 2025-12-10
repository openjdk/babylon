package jdk.incubator.code.runtime;

import jdk.incubator.code.Op;
import java.lang.invoke.CallSite;
import java.lang.invoke.ConstantCallSite;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.NoSuchElementException;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;

public class ReflectableMethodMetafactory {

    private ReflectableMethodMetafactory() {
    }

    /**
     * Generates an implementation of the referenced method's code model to enable
     * dynamic invocation.
     *
     * @param caller Represents a lookup context with the accessibility
     *               privileges of the caller.  Specifically, the lookup context
     *               must have {@linkplain MethodHandles.Lookup#hasFullPrivilegeAccess()
     *               full privilege access}.
     *               When used with {@code invokedynamic}, this is stacked
     *               automatically by the VM.
     * @param methodName The name of the method to implement.  When used with
     *                   {@code invokedynamic}, this is provided by the
     *                   {@code NameAndType} of the {@code InvokeDynamic}
     *                   structure and is stacked automatically by the VM.
     * @param methodType The expected signature of the {@code CallSite}. When
     *                   used with {@code invokedynamic}, this is provided by
     *                   the {@code NameAndType} of the {@code InvokeDynamic}
     *                   structure and is stacked automatically by the VM.
     * @return a CallSite whose target can be used to invoke the method.
     * @throws NoSuchMethodException If a matching method is not found.
     * @throws NoSuchElementException If the method is code model is not present.
     */
    public static CallSite methodInvoke(MethodHandles.Lookup caller,
                                        String methodName,
                                        MethodType methodType) throws NoSuchMethodException {
        CoreOp.FuncOp fop = Op.ofMethod(
                findMethod(caller.lookupClass(),
                          methodName,
                          methodType.parameterArray())).orElseThrow();
        MethodHandle mh = BytecodeGenerator.generate(caller, fop);
        return new ConstantCallSite(mh);
    }

    private static Method findMethod(Class<?> cls, String methodName, Class<?>[] parameters) throws NoSuchMethodException {
        if (parameters.length > 0 && cls == parameters[0]) {
            try {
                // looking for instance method
                Method m = cls.getDeclaredMethod(methodName, Arrays.copyOfRange(parameters, 1, parameters.length));
                if (!m.accessFlags().contains(AccessFlag.STATIC)) {
                    return m;
                }
            } catch (NoSuchMethodException _) {
            }
        }
        // looking for static method
        Method m = cls.getDeclaredMethod(methodName, parameters);
        assert m.accessFlags().contains(AccessFlag.STATIC);
        return m;
    }
}

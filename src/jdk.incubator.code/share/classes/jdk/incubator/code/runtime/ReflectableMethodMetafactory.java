package jdk.incubator.code.runtime;

import java.lang.classfile.ClassFile;
import jdk.incubator.code.Op;
import java.lang.invoke.CallSite;
import java.lang.invoke.ConstantCallSite;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.NoSuchElementException;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;

public class ReflectableMethodMetafactory {

    private ReflectableMethodMetafactory() {
    }

    /**
     * Generates an implementation of the referenced method's code model for
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
    public static CallSite unreflectMethod(MethodHandles.Lookup caller,
                                           String methodName,
                                           MethodType methodType) throws NoSuchMethodException {
        Method[] methods = caller.lookupClass().getDeclaredMethods();
        for (Method m : methods) {
            int rec = (m.getModifiers() & Modifier.STATIC) == 0 ? 1 : 0;
            if (m.getName().matches(methodName)
                    && m.getReturnType() == methodType.returnType()
                    && m.getParameterCount() == methodType.parameterCount() - rec
                    && Arrays.equals(m.getParameterTypes(), 0, m.getParameterCount(),
                                     methodType.parameterArray(), rec, methodType.parameterCount())) {

                CoreOp.FuncOp fop = Op.ofMethod(m).orElseThrow();
                try {
                    MethodHandle mh = BytecodeGenerator.generate(caller, fop);
                    return new ConstantCallSite(mh);
                } catch (Error | Exception e) {
                    System.out.println(fop.toText());
                    throw e;
                }
            }
        }
        throw new NoSuchMethodException(caller.lookupClass().getName() + "." + methodName + methodType);
    }
}

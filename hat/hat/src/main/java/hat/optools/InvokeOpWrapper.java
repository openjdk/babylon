package hat.optools;

import hat.buffer.Buffer;
import hat.buffer.KernelContext;

import java.lang.reflect.Method;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.util.Optional;
import java.util.stream.Stream;

// Is this really a root?
public class InvokeOpWrapper extends OpWrapper<CoreOp.InvokeOp> {
    public InvokeOpWrapper(CoreOp.InvokeOp op) {
        super(op);
    }

    public MethodRef methodRef() {
        return op().invokeDescriptor();
    }

    public JavaType javaRefType() {
        return (JavaType) methodRef().refType();
    }

    public boolean isIfaceBufferMethod() {
        return FuncOpWrapper.ParamTable.Info.isIfaceBuffer(javaRefType());

    }
    public boolean isKernelContextMethod() {
        return FuncOpWrapper.ParamTable.Info.isKernelContext(javaRefType());

    }
    private boolean isReturnTypeAssignableFrom(Class<?> clazz) {
        Optional<Class<?>> optionalClazz = javaReturnClass();
        return optionalClazz.isPresent() && clazz.isAssignableFrom(optionalClazz.get());
    }

    public JavaType javaReturnType() {
        return (JavaType) methodRef().type().returnType();
    }

    public boolean returnsVoid() {
        return javaReturnType().equals(JavaType.VOID);
    }

    public Method method() {
        Class<?> declaringClass = javaRefClass().orElseThrow();
        // TODO this is just matching the name....
        Optional<Method> declaredMethod = Stream.of(declaringClass.getDeclaredMethods())
                .filter(method -> method.getName().equals(methodRef().name()))
                .findFirst();
        if (declaredMethod.isPresent()) {
            return declaredMethod.get();
        }
        Optional<Method> nonDeclaredMethod = Stream.of(declaringClass.getMethods())
                .filter(method -> method.getName().equals(methodRef().name()))
                .findFirst();
        if (nonDeclaredMethod.isPresent()){
            return nonDeclaredMethod.get();
        }else {
            throw new IllegalStateException("what were we looking for ?"); // getClass causes this
            //return nonDeclaredMethod.get();
        }
    }

    public Value getReceiver() {
        return hasReceiver()?operandNAsValue(0):null;
    }
    public boolean hasReceiver() {
        return op().hasReceiver();
    }

    public enum IfaceBufferAccess {None, Access, Mutate}

    public boolean isIfaceAccessor() {
        if (isIfaceBufferMethod() && !returnsVoid()) {
            return !isReturnTypeAssignableFrom(Buffer.class);
        } else {
            return false;
        }
    }

    public boolean isKernelContextAccessor() {
        return isKernelContextMethod();
    }

    public boolean isIfaceMutator() {
        return isIfaceBufferMethod() && returnsVoid();
    }

    public IfaceBufferAccess getIfaceBufferAccess() {
        return isIfaceAccessor() ? IfaceBufferAccess.Access : isIfaceMutator() ? IfaceBufferAccess.Mutate : IfaceBufferAccess.None;
    }


    public String name() {
        return op().invokeDescriptor().name();
    }

    public Optional<Class<?>> javaRefClass() {
        try {
            String className = javaRefType().toString();
            Class<?> javaRefClass = Class.forName(className);
            return Optional.of(javaRefClass);
        } catch (ClassNotFoundException e) {
            return Optional.empty();
        }
    }

    public Optional<Class<?>> javaReturnClass() {
        try {
            String className = javaReturnType().toString();
            Class<?> javaRefClass = Class.forName(className);
            return Optional.of(javaRefClass);
        } catch (ClassNotFoundException e) {
            return Optional.empty();
        }
    }

}

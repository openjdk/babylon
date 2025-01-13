package jdk.incubator.code.internal;

import jdk.incubator.code.Quoted;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.util.Objects;

public class QuotableLambdaOpInvocationHandler implements InvocationHandler {
    private final Object fiInstance;
    private final Quoted quoted;

    public QuotableLambdaOpInvocationHandler(Object fiInstance, Quoted quoted) {
        this.fiInstance = fiInstance;
        this.quoted = quoted;
    }
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        if (Objects.equals(method.getName(), "quoted") && method.getParameterCount() == 0) {
            return quoted();
        } else {
            // Delegate to FI instance
            return method.invoke(fiInstance, args);
        }
    }

    public final Quoted quoted() {
        return quoted;
    }
}

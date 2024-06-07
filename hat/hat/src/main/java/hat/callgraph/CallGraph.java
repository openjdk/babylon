package hat.callgraph;

import hat.ComputeContext;
import hat.optools.FuncOpWrapper;

import java.lang.reflect.Method;
import java.lang.reflect.code.type.MethodRef;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public abstract class CallGraph<E extends Entrypoint> {
    public final ComputeContext computeContext;
    public final E entrypoint;
    public final Set<MethodCall> calls = new HashSet<>();
    public final Map<MethodRef, MethodCall> methodRefToMethodCallMap = new LinkedHashMap<>();

    public Stream<MethodCall> callStream() {
        return methodRefToMethodCallMap.values().stream();
    }

    public interface Resolved {
        FuncOpWrapper funcOpWrapper();

        void funcOpWrapper(FuncOpWrapper funcOpWrapper);
    }

    public interface Unresolved {
    }

    public abstract static class MethodCall {

        public final Method method;
        public final Class<?> declaringClass;
        public CallGraph<?> callGraph;
        public final Set<MethodCall> calls = new HashSet<>();
        public final Set<MethodCall> callers = new HashSet<>();
        public final MethodRef targetMethodRef;

        public boolean closed = false;
        public int rank = 0;

        MethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method) {
            this.callGraph = callGraph;
            this.targetMethodRef = targetMethodRef;
            this.method = method;
            this.declaringClass = method.getDeclaringClass();
        }


        public void dump(String indent) {
            System.out.println(indent + ((targetMethodRef == null ? "EntryPoint" : targetMethodRef)));
            calls.forEach(call -> call.dump(indent + " -> "));
        }


        public void addCall(MethodCall methodCall) {
            callGraph.calls.add(methodCall);
            methodCall.callers.add(this);
            this.calls.add(methodCall);
        }

        protected void rankRecurse(int value) {
            calls.forEach(c -> c.rankRecurse(value + 1));
            if (value > this.rank) {
                this.rank = value;
            }
        }

        public void rankRecurse() {
            rankRecurse(0);
        }
    }

    public abstract static class ResolvedMethodCall extends MethodCall implements Resolved {
        private FuncOpWrapper funcOpWrapper;

        ResolvedMethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
            super(callGraph, targetMethodRef, method);
            this.funcOpWrapper = funcOpWrapper;
        }

        @Override
        public FuncOpWrapper funcOpWrapper() {
            return funcOpWrapper;
        }

        @Override
        public void funcOpWrapper(FuncOpWrapper funcOpWrapper) {
            this.funcOpWrapper = funcOpWrapper;
        }
    }


    public abstract static class UnresolvedMethodCall extends MethodCall implements Unresolved {
        UnresolvedMethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }


    CallGraph(ComputeContext computeContext, E entrypoint) {
        this.computeContext = computeContext;
        this.entrypoint = entrypoint;
    }
}

package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.BiMap;

import java.lang.invoke.MethodHandles;
import optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;

public interface InvokeQuery extends Query<JavaOp.InvokeOp,Invoke,InvokeQuery> {
    record Impl(MethodHandles.Lookup lookup) implements InvokeQuery {
        @Override
        public Res<JavaOp.InvokeOp,Invoke,InvokeQuery> test(CodeElement<?, ?> ce) {
            if (ce instanceof JavaOp.InvokeOp invokeOp) {
                record  MatchImpl (InvokeQuery query, Invoke helper) implements Match<JavaOp.InvokeOp,Invoke,InvokeQuery>{
                    @Override
                    public Match<JavaOp.InvokeOp,Invoke,InvokeQuery> remap(BiMap<CodeElement<?, ?>, CodeElement<?, ?>> biMap) {
                        return null;
                    }
                }
                return new  MatchImpl(this,Invoke.invoke(lookup,invokeOp));
            } else {
                return new Fail<>() {
                };
            }
        }
    }

    static InvokeQuery create(MethodHandles.Lookup lookup) {
         return new Impl(lookup);
    }
}

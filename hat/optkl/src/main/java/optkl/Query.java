package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.BiMap;
import optkl.util.carriers.LookupCarrier;
import optkl.OpHelper.Named.NamedStaticOrInstance.Invoke;
import java.lang.invoke.MethodHandles;

public interface  Query<OH extends OpHelper<?>> extends LookupCarrier {

    interface Res<Query> {
    }
    Res<Query<OH>> test(CodeElement<?,?> ce);

    interface Fail<OH extends OpHelper<?>> extends Res<OH>{
    }
    interface Match<OH extends OpHelper<?>> extends Res<OH>{
        Query<OH> query();
        OH helper();
        Match<OH> remap(BiMap<CodeElement<?,?>, CodeElement<?,?>> biMap);
    }
    interface InvokeQuery<OH extends Invoke> extends Query<OH>{
        record Impl<OH extends Invoke>(Query<OH> query, OH helper) implements Match<OH> {
            @Override
            public Match<OH> remap(BiMap<CodeElement<?, ?>, CodeElement<?, ?>> biMap) {
                return new Impl<>(query, null);
            }
        }
        static <OH extends Invoke>InvokeQuery<OH> create(MethodHandles.Lookup lookup){
            return new InvokeQuery<>() {
                @Override
                public MethodHandles.Lookup lookup(){
                    return lookup;
                }
                @Override
                public Res<Query<OH>> test(CodeElement<?, ?> ce) {
                    if (ce instanceof JavaOp.InvokeOp invokeOp) {
                        Invoke invoke =Invoke.invoke(lookup,invokeOp);
                        return  new Impl<>((InvokeQuery)this,invoke);
                    }else{
                        return new Fail() {};
                    }
                }
            };
        }
    }
}

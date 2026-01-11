package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import optkl.util.BiMap;
import optkl.util.carriers.LookupCarrier;

public interface  Query<O extends Op, OH extends OpHelper<O>, Q extends Query<O,OH,Q>> extends LookupCarrier {

    interface Res<O extends Op, OH extends OpHelper<O>, Q extends Query<O,OH,Q>> {
    }
    Res<O,OH,Q> test(CodeElement<?,?> ce);

    interface Fail<O extends Op, OH extends OpHelper<O>, Q extends Query<O,OH,Q>> extends Res<O,OH,Q>{
    }
    interface Match<O extends Op, OH extends OpHelper<O>, Q extends Query<O,OH,Q>> extends Res<O,OH,Q>{
        Q query();
        OH helper();
        Match<O,OH,Q> remap(BiMap<CodeElement<?,?>, CodeElement<?,?>> biMap);
    }
}

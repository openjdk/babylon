package hat.dialect;

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.ClassType;

import java.util.List;

public abstract class HatMemoryOp extends HatOP {

    public HatMemoryOp(String name, List<Value> operands) {
        super(name, operands);
    }

    protected HatMemoryOp(Op that, CopyContext cc) {
        super(that, cc);
    }

    public String varName() {
        return opName();
    }

    public abstract ClassType classType();

    public abstract TypeElement invokeType();
}

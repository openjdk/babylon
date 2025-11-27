package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Value;

import java.util.List;

public abstract class HATBFloat16Op extends HATOp {

    private String varName;

    public HATBFloat16Op(String varName, List<Value> operands) {
        super(operands);
        this.varName = varName;
    }

    protected HATBFloat16Op(HATBFloat16Op that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
    }

    public String varName() {
        return varName;
    }

    public void varName(String varName) {
        this.varName = varName;
    }
}
